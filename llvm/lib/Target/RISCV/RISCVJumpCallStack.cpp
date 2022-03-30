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
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/ELF.h"
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
#include "llvm/Support/TargetRegistry.h"
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

static cl::opt<bool> ClJumpCallStackAlwaysFixCallers(
    "jump-call-stack-always-fix-callers",
    cl::desc("Always check all callers for a <jump-call-stack-postfix> version "
             "and adjust the call if it is available, even if no functions use "
             "the Jumpoline."),
    cl::Hidden, cl::init(false));

namespace {

enum CallStackMethod {
  JCS_None,
  JCS_Jump,
  JCS_Inline,
};

static CallStackMethod getFunctionCSM(const Function &F) {
  if (F.hasFnAttribute("jump-call-stack")) {
    auto A = F.getFnAttribute("jump-call-stack");
    if (A.getValueAsString().equals("inline"))
      return JCS_Inline;
    if (A.getValueAsString().equals("jump"))
      return JCS_Jump;
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
  bool runFixCallOnMachineFunction(MachineFunction &Fn);
  void runVerifyCallOnMachineFunction(MachineFunction &Fn);
  bool expandCall(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI,
                  MachineOptimizationRemarkEmitter &MORE, bool TailCall);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI,
                MachineOptimizationRemarkEmitter &MORE);
  bool expandMBB(MachineBasicBlock &MBB,
                 MachineOptimizationRemarkEmitter &MORE);

  void insertJCSInline(MachineBasicBlock::iterator &MBBI);
  void insertJCSJump(MachineBasicBlock::iterator &MBBI);
  void insertJCSEpilogue(MachineBasicBlock::iterator &MBBI);
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
    DefinitelyHasPostjump |= CSM == JCS_Jump;
    if (CSM == JCS_None)
      continue;
    bool NewChanged = this->runExpandPEOnMachineFunction(*MaybeMF, CSM);
    Changed |= NewChanged;
  }

  if (DefinitelyHasPostjump || ClJumpCallStackAlwaysFixCallers) {
    for (Function &F : M.functions()) {
      MachineFunction *MaybeMF = MMI.getMachineFunction(F);
      if (!MaybeMF)
        continue;
      bool NewChanged = this->runFixCallOnMachineFunction(*MaybeMF);
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

void RISCVJumpCallStack::insertJCSJump(MachineBasicBlock::iterator &MBBI) {
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

void RISCVJumpCallStack::insertJCSInline(MachineBasicBlock::iterator &MBBI) {
  auto &MBB = *MBBI->getParent();
  auto &MF = *MBB.getParent();
  auto *TII =
      static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  auto &DL = MBBI->getDebugLoc();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();
  Register SPReg = RISCV::X2;
  Register TReg = RISCV::X7;
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  MachineOptimizationRemark R(DEBUG_TYPE, "InlineCallStackSaved", DL, &MBB);
  R << "Inlined RA Save in " << MF.getName();
  MORE.emit(R);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI), TReg)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_HI);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD))
      .addReg(TReg)
      .addReg(TReg)
      .addReg(SPReg);
  BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
      .addReg(RAReg)
      .addReg(TReg)
      .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                         RISCVII::MO_LO);
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
  Register TReg = RISCV::X7;
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

// Fixup prologue and epilogue according to CallStackMethod
bool RISCVJumpCallStack::runExpandPEOnMachineFunction(MachineFunction &MF,
                                                      CallStackMethod CSM) {
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
          insertJCSInline(MBBI);
          break;
        case JCS_Jump:
          insertJCSJump(MBBI);
          break;
        }
      } else if (MBBI->getFlag(MachineInstr::MIFlag::FrameDestroy)) {
        insertJCSEpilogue(MBBI);
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

bool RISCVJumpCallStack::runFixCallOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  for (MachineBasicBlock &MBB : MF) {
    Modified |= expandMBB(MBB, MORE);
  }
  return Modified;
}

bool RISCVJumpCallStack::expandMBB(MachineBasicBlock &MBB,
                                   MachineOptimizationRemarkEmitter &MORE) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI, MORE);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVJumpCallStack::expandMI(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  MachineBasicBlock::iterator &NextMBBI,
                                  MachineOptimizationRemarkEmitter &MORE) {
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoCALL:
    return expandCall(MBB, MBBI, NextMBBI, MORE, false);
    // TODO:  PseudoCALLReg?
  case RISCV::PseudoTAIL:
    return expandCall(MBB, MBBI, NextMBBI, MORE, true);
  }
  return false;
}

bool RISCVJumpCallStack::expandCall(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    MachineBasicBlock::iterator &NextMBBI,
                                    MachineOptimizationRemarkEmitter &MORE,
                                    bool TailCall) {
  static const unsigned OperandNo = 0;
  MachineFunction *MF = MBB.getParent();
  auto *TII =
      static_cast<const RISCVInstrInfo *>(MF->getSubtarget().getInstrInfo());
  MachineInstr &MI = *MBBI;
  MachineOperand &MO = MI.getOperand(OperandNo);
  DebugLoc DL = MI.getDebugLoc();
  StringRef OrigName = getOperandName(MO);
  MCContext &MC = MF->getContext();
  if (OrigName.empty()) {
    MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedEmpty", DL, &MBB);
    SmallString<128> SS;
    raw_svector_ostream OS(SS);
    OS << MO << "(" << (int)MO.getType();
    R << "Not fixing [empty] " << SS << ")";
    MORE.emit(R);
    return false;
  }
  MCSymbol *DestSym = NULL;
  std::string NewName = "";
  if (!OrigName.endswith(ClJumpCallStackPostfix)) {
    NewName = getNewDestinationName(OrigName).str();
    DestSym = MC.lookupSymbol(NewName);
    if (!DestSym) {
      MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedNoDest", DL, &MBB);
      R << "Not fixing [no DestSym=" << NewName << "] from " << OrigName;
      MORE.emit(R);
      // no rewrite symbol; just continue
      return false;
    }
    if (auto *ELFLabel = dyn_cast<MCSymbolELF>(DestSym)) {
      if (ELFLabel->getBinding() == ELF::STB_WEAK) {
        MachineOptimizationRemark R(DEBUG_TYPE, "NotFixedDestWeak", DL, &MBB);
        R << "Ignoring rename for weak binding of " << NewName;
        MORE.emit(R);
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

  auto FirstMI = BuildMI(NewMBB, DL, TII->get(RISCV::AUIPC), RISCV::X6);

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
  BuildMI(NewMBB, DL, TII->get(RISCV::ADDI), RISCV::X6)
      .addReg(RISCV::X6)
      .addMBB(NewMBB, RISCVII::MO_PCREL_LO);
  ((TailCall)
       ? BuildMI(NewMBB, DL, TII->get(RISCV::PseudoJump))
             .addReg(RISCV::X7 /* t2 */, RegState::Define | RegState::Dead)
       : BuildMI(NewMBB, DL, TII->get(RISCV::PseudoCALLReg), RISCV::X1 /* ra */)
             .addReg(RISCV::X6, RegState::ImplicitKill))
      .addExternalSymbol(ClJumpCallStackJumpoline.c_str(), RISCVII::MO_CALL);

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
  switch (MI.getOpcode()) {
  default:
    report_fatal_error("Unsupported Opcode for getJCSPseudoCallSizeInBytes");
  // TODO: PseudoCALLReg?
  case RISCV::PseudoCALL:
  case RISCV::PseudoTAIL: {
    const auto &MO = MI.getOperand(0);
    if (const auto *DestFcn = dyn_cast_or_null<Function>(getGlobalValue(MO))) {
      if (DestFcn->hasFnAttribute("jump-call-stack")) {
        if (DestFcn->getFnAttribute("jump-call-stack")
                .getValueAsString()
                .equals("jump")) {
          Ret = 16;
        }
      }
      // No JCS -- leave at default
      break;
    }
    auto OpName = getOperandName(MO);
    if (OpName.empty())
      break;
    auto &MC = MI.getMF()->getContext();
    if (MC.lookupSymbol(getNewDestinationName(OpName))) {
      Ret = 16;
      break;
    }
    if (MC.lookupSymbol(OpName)) {
      MC.reportWarning(
          SMLoc(),
          "Assuming destination function will not get JCS treatment: " +
              OpName);
      Ret = 8;
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
  return  CSM == JCS_Jump || CSM == JCS_Inline;
}