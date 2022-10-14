//===-- RISCVJumpCallStack.cpp - Handle optimizations for JumpCallStack ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handles Prolog and Epilog changes for JumpCallStack and InlineCallStack
// sanitizers
//
// Convert calls to JumpCallStack functions to go directly to the jumpoline.
// This pass transforms:
//   call function
//
//   Into:
//   la t1, function$postjump
//   call ra, tcb_jumpoline
//
// The transformation is carried out under certain conditions:
// 1) The destination function has the JumpCallStack attribute
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Metadata.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-optimize-jump-call-stack"
#define RISCV_JUMP_CALL_STACK_OPTIMIZE_NAME "RISCV Jump Call Stack optimization"

static cl::opt<std::string> ClJumpCallStackPostfix(
    "jump-call-stack-postfix",
    cl::desc("When using JumpCallStack: Suffix for labeling the part of the "
             "function after the prologue jump"),
    cl::Hidden, cl::init("$postjump"));

static cl::opt<std::string> ClJumpInlineCallStackPointerOffset(
    "jump-call-stack-pointer-offset",
    cl::desc("When using JumpCallStack and InlineCallStack: Label of the "
             "offset to be added to the stack pointer to find the appropriate "
             "location in the secure stack"),
    cl::Hidden, cl::init("__tcb_sp_offset"));

static cl::opt<std::string> ClJumpCallStackJumpoline(
    "jump-call-stack-jumpoline",
    cl::desc(
        "When using JumpCallStack: label of the function to call that will "
        "save the return address and jump to the destination function"),
    cl::Hidden, cl::init("tcb_jumpoline"));

static cl::opt<std::string> ClJumpCallStackJumpolinePop(
    "jump-call-stack-jumpoline-pop",
    cl::desc("When using JumpCallStack: label of the function to call that "
             "will save the return address and return to the prolog from which "
             "it was called"),
    cl::Hidden, cl::init("tcb_jumpoline_pop"));

static cl::opt<std::string> ClJumpCallStackJumpolineTail(
    "jump-call-stack-jumpoline-tail",
    cl::desc("When using JumpCallStack: label of the function to call that "
             "will save the return address and return to the prolog from which "
             "it was called"),
    cl::Hidden, cl::init("tcb_jumpoline_tail"));

static cl::opt<bool> ClJumpCallStackAlwaysFixCallers(
    "jump-call-stack-always-fix-callers",
    cl::desc("Always check all callers for a <jump-call-stack-postfix> version "
             "and adjust the call if it is available, even if no functions use "
             "the Jumpoline."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClJumpCallStackOverflowCheck(
    "jump-call-stack-add-overflow-check",
    cl::desc(
        "Add a stack overflow check in the form of sw zero, -(XLEN/8)(sp) to "
        "ensure that the TCB cannot overflow the stack."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClJumpCallStackForce8byteSPOffset(
    "jump-call-stack-force-64b-sp-offset",
    cl::desc("Force the use assumption of XLEN=64 for "
             "jump-call-stack-add-overflow-check"),
    cl::Hidden, cl::init(true));
// Default tcb.ld has this assumption because ldscript doesn't know any better

static cl::opt<bool> ClJumpCallStackAlwaysScan(
    "jump-call-stack-always-scan",
    cl::desc("Always perform code scanning for calls to "
             "<jump-call-stack-jumpoline> and <jump-call-stack-jumpoline-pop> "
             "even if no functions use the jumpoline"),
    cl::Hidden, cl::init(false));

namespace {

enum CallStackMethod {
  JCS_None,
  JCS_Jump,
  JCS_Inline,
  JCS_JumpCompressed,
};

static CallStackMethod getFunctionCSM(const Function &F) {
  if (F.hasFnAttribute("jump-call-stack")) {
    auto A = F.getFnAttribute("jump-call-stack");
    if (A.getValueAsString().equals("inline"))
      return JCS_Inline;
    if (A.getValueAsString().equals("jump"))
      return JCS_Jump;
    if (A.getValueAsString().equals("jumpCompressed"))
      return JCS_JumpCompressed;
    errs() << A.getValueAsString() << "\n";
    report_fatal_error("Unable to handle jump-call-stack type");
  }
  return JCS_None;
}

struct RISCVJumpCallStack : public ModulePass {
  static char ID;
  bool runOnModule(Module &M) override;
  RISCVJumpCallStack() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return RISCV_JUMP_CALL_STACK_OPTIMIZE_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addPreserved<MachineModuleInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  bool runExpandPEOnMachineFunction(MachineFunction &Fn, CallStackMethod CSM);
  bool runFixCallOnMachineFunction(MachineFunction &Fn, bool DoFixCallers);
  void runVerifyCallOnMachineFunction(MachineFunction &Fn);
  bool expandCall(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI,
                  MachineOptimizationRemarkEmitter &MORE,
                  DenseMap<int, MachineBasicBlock::iterator> &MBBNextI,
                  bool TailCall);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI,
                MachineOptimizationRemarkEmitter &MORE,
                DenseMap<int, MachineBasicBlock::iterator> &MBBNextI,
                bool DoFixCallers);
  bool expandMBB(MachineBasicBlock &MBB, MachineOptimizationRemarkEmitter &MORE,
                 DenseMap<int, MachineBasicBlock::iterator> &MBBNextI,
                 bool DoFixCallers);

  void insertJCSPrologueInline(MachineBasicBlock::iterator &MBBI);
  void insertJCSPrologueJump(MachineBasicBlock::iterator &MBBI);
  void insertJCSEpilogue(MachineBasicBlock::iterator &MBBI);
  void insertJCSEpilogueCompressed(MachineBasicBlock::iterator &MBBI);
};

} // end anonymous namespace

static const GlobalValue *getGlobalValue(const MachineOperand &MO) {
  if (MO.isGlobal()) {
    return MO.getGlobal();
  }
  return nullptr;
}

static StringRef getOperandName(const MachineOperand &MO,
                                bool AllowNonzeroOffset = false) {
  if (!(MO.isGlobal() || MO.isSymbol()))
    return StringRef();
  if (AllowNonzeroOffset || MO.getOffset() == 0) {
    if (const GlobalValue *GV = getGlobalValue(MO))
      return GV->getName();
    if (MO.isSymbol())
      return MO.getSymbolName();
  }
  return StringRef();
}

static llvm::Twine getNewDestinationName(StringRef OrigName) {
  return (OrigName + ClJumpCallStackPostfix);
}

char RISCVJumpCallStack::ID = 0;
INITIALIZE_PASS(RISCVJumpCallStack, DEBUG_TYPE,
                RISCV_JUMP_CALL_STACK_OPTIMIZE_NAME, false, false)

bool RISCVJumpCallStack::runOnModule(Module &M) {
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  bool Changed = false;
  bool DefinitelyHasPostjump = false;

  for (Function &F : M.functions()) {
    MachineFunction *MaybeMF = MMI.getMachineFunction(F);
    if (!MaybeMF)
      continue;
    CallStackMethod CSM = getFunctionCSM(F);
    DefinitelyHasPostjump |= (CSM == JCS_Jump || CSM == JCS_JumpCompressed);
    if (CSM == JCS_None)
      continue;
    bool NewChanged = this->runExpandPEOnMachineFunction(*MaybeMF, CSM);
    Changed |= NewChanged;
  }

  bool DoFixCallers = DefinitelyHasPostjump || ClJumpCallStackAlwaysFixCallers;

  if (DoFixCallers || ClJumpCallStackAlwaysScan) {
    for (Function &F : M.functions()) {
      MachineFunction *MaybeMF = MMI.getMachineFunction(F);
      if (!MaybeMF)
        continue;
      bool NewChanged =
          this->runFixCallOnMachineFunction(*MaybeMF, DoFixCallers);
      Changed |= NewChanged;
    }
  }

  return Changed;
}

static bool findInstrumentNOP(MachineInstr &MI) {
  for (auto *MOI = MI.operands_begin(); MOI != MI.operands_end(); MOI++) {
    if (MOI->isMetadata()) {
      auto *MDN = MOI->getMetadata();
      for (const MDOperand &MDO : MDN->operands()) {
        if (auto *MDS = dyn_cast_or_null<MDString>(MDO.get())) {
          if (MDS->getString().equals("JumpCallStackInlineCallStack")) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

static bool findInstrumentNOP(MachineInstr &MI, uint_least16_t Flags) {
  auto MIFlags = MI.getFlags();
  static_assert(sizeof(Flags) >= sizeof(MIFlags), "Flags too small");
  if (MIFlags & Flags || !Flags)
    return findInstrumentNOP(MI);
  return false;
}

static MachineBasicBlock::iterator findInstrumentNOP(MachineBasicBlock &MBB,
                                                     uint_least16_t Flags) {
  for (auto MBBI = MBB.begin(); MBBI != MBB.end(); MBBI++) {
    if (findInstrumentNOP(*MBBI, Flags)) {
      return MBBI;
    }
  }
  return MBB.end();
}

void RISCVJumpCallStack::insertJCSPrologueJump(
    MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  auto *TII =
      static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  auto &F = MF.getFunction();
  auto &DL = MBBI->getDebugLoc();
  if (!(F.hasExternalLinkage() || F.hasAddressTaken() || F.isWeakForLinker())) {
    MachineOptimizationRemark R(DEBUG_TYPE, "JumpCallStackRenamed", DL, &MBB);
    R << "Renamed " << MF.getName() << " because all callers use jumpoline";
    MORE.emit(R);
    F.setName(getNewDestinationName(MF.getName()));
  } else {
    // Store return address to jump call stack
    // call t0, __tcb_jumpoline_pop
    MachineOptimizationRemark R(DEBUG_TYPE, "JumpCallStackAdded", DL, &MBB);
    R << "Added jumpoline jump and label to " << MF.getName();
    MORE.emit(R);
    const auto &STI = MF.getSubtarget<RISCVSubtarget>();
    bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
    if (ClJumpCallStackOverflowCheck) {
      int64_t SlotSize =
          ClJumpCallStackForce8byteSPOffset ? 8 : STI.getXLen() / 8;
      BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
          .addReg(RISCV::X0)
          .addReg(RISCV::X2 /* sp */)
          .addImm(-SlotSize)
          .setMIFlag(MachineInstr::FrameSetup);
    }
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoCALLReg))
        .addReg(RISCV::X5)
        .addExternalSymbol(ClJumpCallStackJumpolinePop.c_str(),
                           RISCVII::MO_CALL)
        .setMIFlag(MachineInstr::FrameSetup);
    MCSymbol *Label =
        MF.getContext().getOrCreateSymbol(getNewDestinationName(MF.getName()));
    if (MF.getFunction().hasExternalLinkage()) {
      if (auto *ELFLabel = dyn_cast<MCSymbolELF>(Label)) {
        if (MF.getFunction().isWeakForLinker()) {
          ELFLabel->setBinding(ELF::STB_WEAK);
        } else {
          ELFLabel->setBinding(ELF::STB_GLOBAL);
        }
      }
    }
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::EH_LABEL))
        .addSym(Label)
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void RISCVJumpCallStack::insertJCSPrologueInline(
    MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  auto *TII =
      static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  auto &DL = MBBI->getDebugLoc();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();
  Register SPReg = RISCV::X2;
  Register TReg = RISCV::X5; // t0
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  MachineOptimizationRemark R(DEBUG_TYPE, "InlineCallStackSaved", DL, &MBB);
  R << "Inlined RA Save in " << MF.getName();
  MORE.emit(R);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI), TReg)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_HI)
      .setMIFlag(MachineInstr::FrameSetup);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD))
      .addReg(TReg)
      .addReg(TReg)
      .addReg(SPReg)
      .setMIFlag(MachineInstr::FrameSetup);
  BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
      .addReg(RAReg)
      .addReg(TReg)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_LO)
      .setMIFlag(MachineInstr::FrameSetup);
}

void RISCVJumpCallStack::insertJCSEpilogue(MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  auto *TII =
      static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  auto &DL = MBBI->getDebugLoc();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();
  Register SPReg = RISCV::X2;
  Register TReg = RISCV::X5; // t0
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  MachineOptimizationRemark R(DEBUG_TYPE, "JumpCallStackRestored", DL, &MBB);
  R << "Added RA restore to " << MF.getName();
  MORE.emit(R);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI), TReg)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_HI)
      .setMIFlag(MachineInstr::FrameDestroy);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), TReg)
      .addReg(TReg)
      .addReg(SPReg)
      .setMIFlag(MachineInstr::FrameDestroy);
  BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::LD : RISCV::LW), RAReg)
      .addReg(TReg, RegState::Kill)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_LO)
      .setMIFlag(MachineInstr::FrameDestroy);
}

static void insertSCSPRestore(MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  auto *TII = static_cast<const RISCVInstrInfo *>(STI.getInstrInfo());
  auto &DL = MBBI->getDebugLoc();
  Register const SCSPReg = RISCVABI::getSCSPReg();
  const bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), SCSPReg)
      .addReg(SCSPReg)
      .addImm(IsRV64 ? -8 : -4)
      .setMIFlag(MachineInstr::FrameDestroy);

void RISCVJumpCallStack::insertJCSEpilogueCompressed(
    MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  auto *TII = static_cast<const RISCVInstrInfo *>(STI.getInstrInfo());
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  auto &DL = MBBI->getDebugLoc();
  Register const RAReg = STI.getRegisterInfo()->getRARegister();
  Register const SCSPReg = RISCVABI::getSCSPReg();
  bool const IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  MachineOptimizationRemark R(DEBUG_TYPE, "JumpShadowCallStackRestored", DL,
                              &MBB);
  R << "Added RA restore to " << MF.getName();
  MORE.emit(R);
  /* REG_L ra, -4(x18) */
  BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::LD : RISCV::LW), RAReg)
      .addReg(SCSPReg)
      .addImm(IsRV64 ? -8 : -4)
      .setMIFlag(MachineInstr::FrameDestroy);
  /* addi x18, x18, -SZREG(ra) */
  insertSCSPRestore(MBBI);
}

// Fixup prologue and epilogue according to CallStackMethod
bool RISCVJumpCallStack::runExpandPEOnMachineFunction(MachineFunction &MF,
                                                      CallStackMethod CSM) {
  auto *RI = MF.getSubtarget().getRegisterInfo();
  auto &Ctx = MF.getFunction().getContext();
  switch (CSM) {
  default:
    report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
    break;
  case JCS_Jump:
  case JCS_Inline:
    if (RI->hasStackRealignment(MF) || MF.getFrameInfo().hasVarSizedObjects()) {
      errs() << "Function '" << MF.getName() << "' has";
      if (RI->hasStackRealignment(MF))
        errs() << " Stack Realignment";
      if (RI->hasStackRealignment(MF) && MF.getFrameInfo().hasVarSizedObjects())
        errs() << " and";
      if (MF.getFrameInfo().hasVarSizedObjects())
        errs() << " Variable-sized Objects";
      errs() << "\n";
      MF.print(errs());
      MF.getContext().reportError(
          SMLoc(), MF.getName() + " uses JumpCallStack/InlineCallStack, but "
                                  "modifies the stack pointer");
    }
    break;
  case JCS_JumpCompressed:
    const auto &STI = MF.getSubtarget<RISCVSubtarget>();
    Register const SCSPReg = RISCVABI::getSCSPReg();
    if (!STI.isRegisterReservedByUser(SCSPReg)) {
      Ctx.diagnose(DiagnosticInfoUnsupported{
          MF.getFunction(), "x18 not reserved by user for Shadow Call Stack.",
          MF.begin()->begin()->getDebugLoc()});
      return false;
    }
  }
  bool Modified = false;
  auto MFI = MF.begin();
  for (; (MFI != MF.end()); MFI++) {
    auto &MBB = *MFI;
    auto GetMBBI = [&] {
      return findInstrumentNOP(MBB, MachineInstr::MIFlag::FrameSetup |
                                        MachineInstr::MIFlag::FrameDestroy);
    };
    for (MachineBasicBlock::iterator MBBI = GetMBBI(); MBBI != MBB.end();
         MBBI = GetMBBI()) {
      if (MBBI->getFlag(MachineInstr::MIFlag::FrameSetup)) {
        switch (CSM) {
        default:
          report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
          break;
        case JCS_Inline:
          insertJCSPrologueInline(MBBI);
          break;
        case JCS_Jump:
        case JCS_JumpCompressed:
          if (MBBI->getParent() != &MF.front())
            Ctx.diagnose(DiagnosticInfoUnsupported{
                MF.getFunction(), "RISCVJumpCallStack: Prologue not first MBB",
                MBBI->getDebugLoc(), DS_Error});
          // report_fatal_error("RISCVJumpCallStack: Prologue not first MBB");
          insertJCSPrologueJump(MBBI);
          break;
        }
      } else if (MBBI->getFlag(MachineInstr::MIFlag::FrameDestroy)) {
        switch (CSM) {
        default:
          report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
          break;
        case JCS_Jump:
        case JCS_Inline:
          insertJCSEpilogue(MBBI);
          break;
        case JCS_JumpCompressed:
          insertJCSEpilogueCompressed(MBBI);
        }
      } else {
        report_fatal_error("RISCVJumpCallStack: Got a "
                           "JumpCallStackInlineCallStack without a known flag");
      }
      MBBI->eraseFromParent();
      Modified = true;
    }
  }
  return Modified;
}

bool RISCVJumpCallStack::runFixCallOnMachineFunction(MachineFunction &MF,
                                                     bool DoFixCallers) {
  bool Modified = false;
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  DenseMap<int, MachineBasicBlock::iterator> MBBNextI;
  for (MachineBasicBlock &MBB : MF) {
    Modified |= expandMBB(MBB, MORE, MBBNextI, DoFixCallers);
  }
  return Modified;
}

bool RISCVJumpCallStack::expandMBB(
    MachineBasicBlock &MBB, MachineOptimizationRemarkEmitter &MORE,
    DenseMap<int, MachineBasicBlock::iterator> &MBBNextI, bool DoFixCallers) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  auto WantedMBBI = MBBNextI.find(MBB.getNumber());
  if (WantedMBBI != MBBNextI.end())
    MBBI = WantedMBBI->getSecond();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI, MORE, DoFixCallers);
    Modified |= expandMI(MBB, MBBI, NMBBI, MORE, MBBNextI, DoFixCallers);
    MBBI = NMBBI;
  }

  return Modified;
}

static void checkLoadJumpoline(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI) {
  static const unsigned OperandNo = 1;
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  MachineOperand &MO = MI.getOperand(OperandNo);
  StringRef OrigName = getOperandName(MO);
  if (OrigName.equals(ClJumpCallStackJumpoline) ||
      OrigName.equals(ClJumpCallStackJumpolinePop)) {
    MF->getContext().reportError(SMLoc(), "trampoline address loaded in " +
                                              MF->getName());
  }
}

static void checkInlineASM(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  for (unsigned OperandNo = 1; OperandNo < MBBI->getNumOperands();
       OperandNo++) {
    MachineOperand &MO = MI.getOperand(OperandNo);
    StringRef OrigName = getOperandName(MO);
    if (OrigName.equals(ClJumpCallStackJumpoline) ||
        OrigName.equals(ClJumpCallStackJumpolinePop)) {
      MF->getContext().reportError(SMLoc(), "trampoline address used in " +
                                                MF->getName());
      // report_fatal_error("Somehow returned from emitError");
    }
  }
}

static void checkJumpTail(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI) {
  static const unsigned OperandNo = 0;
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  MachineOperand &MO = MI.getOperand(OperandNo);
  StringRef OrigName = getOperandName(MO);
  if (OrigName.equals(ClJumpCallStackJumpoline) ||
      OrigName.equals(ClJumpCallStackJumpolineTail) || 
      OrigName.equals(ClJumpCallStackJumpolinePop)) {
    MF->getContext().reportError(
        SMLoc(), "jump to a trampoline in (checkJumpTail)" + MF->getName());
  }
}

bool RISCVJumpCallStack::expandMI(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI,
    MachineOptimizationRemarkEmitter &MORE,
    DenseMap<int, MachineBasicBlock::iterator> &MBBNextI, bool DoFixCallers) {
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoCALL:
    if (DoFixCallers)
      return expandCall(MBB, MBBI, NextMBBI, MORE, MBBNextI, false);
    checkJumpTail(MBB, MBBI);
    break;
    // TODO:  PseudoCALLReg?
  case RISCV::PseudoTAIL:
    if (DoFixCallers)
      return expandCall(MBB, MBBI, NextMBBI, MORE, MBBNextI, true);
    checkJumpTail(MBB, MBBI);
    break;
  case RISCV::PseudoLA:
  case RISCV::PseudoLLA:
    checkLoadJumpoline(MBB, MBBI);
    break;
  case RISCV::INLINEASM:
  case RISCV::INLINEASM_BR:
    checkInlineASM(MBB, MBBI);
    break;
  }
  return false;
}

static void reportCallToBadFromJCS(MachineFunction &MF, MachineInstr &MI,
                                   StringRef Dest, StringRef DestProblem) {
  MF.getFunction().getContext().diagnose(
      DiagnosticInfoUnsupported{MF.getFunction(),
                                Twine("Call to ") + DestProblem + " '" + Dest +
                                    "' may corrupt control-flow",
                                MI.getDebugLoc(), DS_Warning});
static void reportCallToBadFromJCS(MachineFunction &MF, StringRef Dest,
                                   StringRef DestProblem) {
  errs() << "JumpCallStack: Call from jump-call-stack function '"
         << MF.getName() << "' to " << DestProblem << " destination '" << Dest
         << "' may corrupt control-flow.\n";
}

bool RISCVJumpCallStack::expandCall(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI,
    MachineOptimizationRemarkEmitter &MORE,
    DenseMap<int, MachineBasicBlock::iterator> &MBBNextI, bool TailCall) {
  static const unsigned OperandNo = 0;
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<RISCVSubtarget>();
  auto *TII = static_cast<const RISCVInstrInfo *>(STI.getInstrInfo());
  MachineInstr &MI = *MBBI;
  MachineOperand &MO = MI.getOperand(OperandNo);
  DebugLoc DL = MI.getDebugLoc();
  StringRef OrigName = getOperandName(MO);
  MCContext &MC = MF->getContext();
  if (MI.getFlag(MachineInstr::FrameSetup)) {
    MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedFrameSetup", DL, &MBB);
    R << "Not fixing [frame setup] (" << OrigName << ")";
    MORE.emit(R);
    return false;
  }
  if (OrigName.empty()) {
    MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedEmpty", DL, &MBB);
    SmallString<128> SS;
    raw_svector_ostream OS(SS);
    OS << MO << "(" << (int)MO.getType();
    R << "Not fixing [empty] " << SS << ")";
    MORE.emit(R);
    reportCallToBadFromJCS(*MF, MI, "[empty]", "unknown");
    return false;
  }
  if (OrigName.equals(ClJumpCallStackJumpoline) ||
      OrigName.equals(ClJumpCallStackJumpolinePop)) {
    MF->getContext().reportError(
        SMLoc(), "jump to a trampoline in (expandCall)" + MF->getName());
    // MI.emitError("jumps to a trampoline");
    // report_fatal_error("Somehow returned from emitError");
  }
  MCSymbol *DestSym = NULL;
  std::string NewName = "";
  if (!OrigName.endswith(ClJumpCallStackPostfix)) {
    NewName = getNewDestinationName(OrigName).str();
    DestSym = MC.lookupSymbol(NewName);
    if (!DestSym) {
      if (auto *Module = MF->getFunction().getParent()) {
        auto *DestFunction = Module->getFunction(OrigName);
        if (!DestFunction) {
          while (auto *DestAlias = Module->getNamedAlias(OrigName)) {
            auto NewAliasName = DestAlias->getAliasee()->getName();
            MachineOptimizationRemark R(DEBUG_TYPE, "FollowingAlias", DL, &MBB);
            R << "Following alias from " << OrigName << " to " << NewAliasName;
            MORE.emit(R);
            OrigName = NewAliasName;
          }
          DestFunction = Module->getFunction(OrigName);
          NewName = getNewDestinationName(OrigName).str();
          DestSym = MC.lookupSymbol(NewName);
        }
        if (DestFunction) {
          if (getFunctionCSM(*DestFunction) == JCS_Jump) {
            MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedDestLeaf", DL,
                                        &MBB);
            R << "Not fixing [no DestSym=" << NewName << ", likely leaf] from "
              << OrigName;
            MORE.emit(R);
            return false;
          }
        }
      }
    }
    if (!DestSym) {
      MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedNoDest", DL, &MBB);
      R << "Not fixing [no DestSym=" << NewName << "] from " << OrigName;
      MORE.emit(R);
      reportCallToBadFromJCS(*MF, MI, OrigName, "non-JCS");
      // no rewrite symbol; just continue
      return false;
    }
    if (auto *ELFLabel = dyn_cast<MCSymbolELF>(DestSym)) {
      if (ELFLabel->getBinding() == ELF::STB_WEAK) {
        MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedDestWeak", DL, &MBB);
        R << "Ignoring rename for weak binding of " << NewName;
        MORE.emit(R);
        reportCallToBadFromJCS(*MF, MI, OrigName, "weak");
        return false;
      }
    }
  }
  if (!DestSym) {
    if (auto *DestF = dyn_cast<Function>(MO.getGlobal())) {
      if (DestF->isWeakForLinker()) {
        DestF->getContext().diagnose(DiagnosticInfoUnsupported(
            *DestF,
            "Why did we rename weak function " + DestF->getName() + "???"));
      }
    }
  }

  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  // Tell AsmPrinter that we unconditionally want the symbol of this label
  // to be emitted.
  NewMBB->setLabelMustBeEmitted();

  MF->insert(++MBB.getIterator(), NewMBB);

  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  int64_t SlotSize = ClJumpCallStackForce8byteSPOffset ? 8 : STI.getXLen() / 8;

  Register PostJumpAddrReg =
      (TailCall) ? RISCV::X5 /* t0 */ : RISCV::X6 /* t1 */;
  Register TailJumpRegisterTemporary = RISCV::X6 /* t1 */;

  /* 1: luipc t1, %pcrel_hi(original_label/value) */
  auto FirstMI = BuildMI(NewMBB, DL, TII->get(RISCV::AUIPC), PostJumpAddrReg);

  if (DestSym) {
    // Ensure that DestSym has a null-terminated name
    assert(strnlen(DestSym->getName().data(), NewName.length() + 1) ==
           NewName.length());

    FirstMI.addExternalSymbol(DestSym->getName().data(), RISCVII::MO_PCREL_HI);
    MachineOptimizationRemark R(DEBUG_TYPE, "FixedLabel", DL, &MBB);
    R << "Fixing reference from " << OrigName << " to " << NewName;
    MORE.emit(R);
  } else {
    // Copy old operand
    FirstMI.addGlobalAddress(MO.getGlobal(), 0, RISCVII::MO_PCREL_HI);
    MachineOptimizationRemark R(DEBUG_TYPE, "FixedValue", DL, &MBB);
    R << "Fixing reference for " << OrigName;
    MORE.emit(R);
  }
  /* addi t1, t1, %pcrel_lo(1b) */
  BuildMI(NewMBB, DL, TII->get(RISCV::ADDI), PostJumpAddrReg)
      .addReg(PostJumpAddrReg)
      .addMBB(NewMBB, RISCVII::MO_PCREL_LO);
  if (ClJumpCallStackOverflowCheck) {
    /* REG_S zero, -SlotSize(sp) */
    BuildMI(NewMBB, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
        .addReg(RISCV::X0)
        .addReg(RISCV::X2 /* sp */)
        .addImm(-SlotSize);
  }
  ((TailCall)
       /* jump tcb_jumpoline */
       ? BuildMI(NewMBB, DL, TII->get(RISCV::PseudoJump))
             .addReg(TailJumpRegisterTemporary,
                     RegState::Define | RegState::Dead)
       /* call ra, tcb_jumpoline */
       : BuildMI(NewMBB, DL, TII->get(RISCV::PseudoCALL)))
      .addExternalSymbol((TailCall) ? ClJumpCallStackJumpolineTail.c_str()
                                    : ClJumpCallStackJumpoline.c_str(),
                         RISCVII::MO_CALL);

  // Save point in the new MBB that's already expanded
  auto NewMBBNextMBBI = std::prev(NewMBB->end());

  // Move all the rest of the instructions to NewMBB.
  NewMBB->splice(NewMBB->end(), &MBB, std::next(MBBI), MBB.end());
  // Update machine-CFG edges.
  NewMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  // Make the original basic block fall-through to the new.
  MBB.addSuccessor(NewMBB);

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *NewMBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  MBBNextI.insert(
      std::make_pair(NewMBB->getNumber(), std::next(NewMBBNextMBBI)));

  return true;
}

static inline void checkOpName(StringRef OpName, MachineInstr &MI,
                               MachineOptimizationRemarkEmitter &MORE) {
  if (OpName.endswith(ClJumpCallStackPostfix) || OpName.empty()) {
    MachineOptimizationRemark R(DEBUG_TYPE, "FailedCallRename",
                                MI.getDebugLoc(), MI.getParent());
    R << "Failed to rename call: " << OpName;
    MORE.emit(R);
  }
}

void RISCVJumpCallStack::runVerifyCallOnMachineFunction(MachineFunction &MF) {
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (findInstrumentNOP(MI)) {
        errs() << MF.getName() << "\n";
        MBB.print(errs());
        MI.print(errs());
        report_fatal_error("Did not remove instrumented NOP");
      }
      switch (MI.getOpcode()) {
      case RISCV::PseudoCALL:
      case RISCV::PseudoTAIL:
        checkOpName(getOperandName(MI.getOperand(0)), MI, MORE);
        break;
      case RISCV::PseudoJump:
        checkOpName(getOperandName(MI.getOperand(1)), MI, MORE);
      }
    }
  }
}

/// Returns an instance of the Merge Base Offset Optimization pass.
ModulePass *llvm::createRISCVJumpCallStackPass() {
  return new RISCVJumpCallStack();
}

unsigned llvm::getJCSPseudoCallSizeInBytes(const MachineInstr &MI) {
  unsigned Ret = 8;
  bool IsRV64 = false;

  const RISCVInstrInfo *TII = nullptr;
  if (const auto *MF = MI.getMF()) {
    const auto &STI = MF->getSubtarget<RISCVSubtarget>();
    IsRV64 = STI.is64Bit();
    TII = static_cast<const RISCVInstrInfo *>(STI.getInstrInfo());
    Ret = TII->get(MI.getOpcode()).getSize();
  }

  switch (MI.getOpcode()) {
  default:
    report_fatal_error("Unsupported Opcode for getJCSPseudoCallSizeInBytes");
  // TODO: PseudoCALLReg?
  case RISCV::PseudoCALL:
  case RISCV::PseudoTAIL: {
    const auto &MO = MI.getOperand(0);
    if (const auto *DestFcn = dyn_cast_or_null<Function>(getGlobalValue(MO))) {
      auto DestFcnCSM = getFunctionCSM(*DestFcn);
      switch (DestFcnCSM) {
      case JCS_Jump:
      case JCS_JumpCompressed:
        if (TII)
          Ret = TII->get(RISCV::PseudoCALL).getSize() +
                TII->get(RISCV::PseudoLA).getSize() +
                (ClJumpCallStackOverflowCheck
                     ? TII->get(IsRV64 ? RISCV::SD : RISCV::SW).getSize()
                     : 0);
        else
          Ret = (ClJumpCallStackOverflowCheck) ? 20 : 16;
        break;
      default:
        break;
      }
      break;
    }
    auto OpName = getOperandName(MO);
    if (OpName.empty())
      break;
    auto &MC = MI.getMF()->getContext();
    if (MC.lookupSymbol(getNewDestinationName(OpName))) {
      if (TII)
        Ret = TII->get(RISCV::PseudoCALL).getSize() +
              TII->get(RISCV::PseudoLA).getSize() +
              (ClJumpCallStackOverflowCheck
                   ? TII->get(IsRV64 ? RISCV::SD : RISCV::SW).getSize()
                   : 0);
      else
        Ret = (ClJumpCallStackOverflowCheck) ? 20 : 16;
      break;
    }
    if (MC.lookupSymbol(OpName)) {
      MC.reportWarning(
          SMLoc(),
          "Assuming destination function will not get JCS treatment: " +
              OpName);
      break;
    }
    // MC.reportWarning(SMLoc(), "Unable to find destination function: " +
    // OpName);
    Ret = 16;
    break;
  }
  }
  return Ret;
}

bool llvm::getJCSFunctionUsesT2(const MachineFunction &MF) {
  auto const CSM = getFunctionCSM(MF.getFunction());
  // Note that the compressed variants do not need t2.
  switch (CSM) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
  default:
#pragma clang diagnostic pop
    report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
    break;
  case JCS_Jump:
  case JCS_Inline:
    return true;
  case JCS_JumpCompressed:
  case JCS_None:
    return false;
  }
}

bool llvm::canJCSFunctionUseShrinkWrap(const MachineFunction &MF) {
  auto const CSM = getFunctionCSM(MF.getFunction());
  // Note that the inline variants never get $postjump optimized, and can use
  // shrink wrap without any problems.
  switch (CSM) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
  default:
#pragma clang diagnostic pop
    report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
    break;
  case JCS_Jump:
  case JCS_JumpCompressed:
    // If we do not save return address, we can shrinkwrap, but probably don't
    // know that yet
    // {
    //   const auto &STI = MF.getSubtarget<RISCVSubtarget>();
    //   Register RAReg = STI.getRegisterInfo()->getRARegister();
    //   auto &CSI = MF.getFrameInfo().getCalleeSavedInfo();
    //   return std::none_of(CSI.begin(), CSI.end(), [&](CalleeSavedInfo &CSR) {
    //     return CSR.getReg() == RAReg;
    //   });
    // }
    return false;
  case JCS_Inline:
  case JCS_None:
    return true;
  }
}
