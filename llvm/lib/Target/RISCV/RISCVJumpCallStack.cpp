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

namespace {

struct RISCVJumpCallStack : public ModulePass {
  const RISCVInstrInfo *TII;
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
  enum CallStackMethod {
    JCS_None,
    JCS_Jump,
    JCS_Inline,
  };

  CallStackMethod getFunctionCSM(Function &F) {
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

  bool runFixLabelOnMachineFunction(MachineFunction &Fn, CallStackMethod CSM);
  bool runFixCallOnMachineFunction(MachineFunction &Fn);
  void runVerifyCallOnMachineFunction(MachineFunction &Fn);
  bool expandCall(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI,
                  MachineOptimizationRemarkEmitter &MORE);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI,
                MachineOptimizationRemarkEmitter &MORE);
  bool expandMBB(MachineBasicBlock &MBB,
                 MachineOptimizationRemarkEmitter &MORE);
};

} // end anonymous namespace

static StringRef getOperandName(MachineOperand &MO,
                                bool AllowNonzeroOffset = false) {
  if (MO.isGlobal()) {
    if (AllowNonzeroOffset || MO.getOffset() == 0) {
      const GlobalValue *GV = MO.getGlobal();
      if (GV)
        return GV->getName();
    }
  } else if (MO.isSymbol()) {
    if (AllowNonzeroOffset || MO.getOffset() == 0) {
      return MO.getSymbolName();
    }
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

  for (Function &F : M.functions()) {
    MachineFunction *MaybeMF = MMI.getMachineFunction(F);
    if (!MaybeMF)
      continue;
    CallStackMethod CSM = getFunctionCSM(F);
    if (CSM == JCS_None)
      continue;
    bool NewChanged = this->runFixLabelOnMachineFunction(*MaybeMF, CSM);
    Changed |= NewChanged;
  }

  for (Function &F : M.functions()) {
    MachineFunction *MaybeMF = MMI.getMachineFunction(F);
    if (!MaybeMF)
      continue;
    bool NewChanged = this->runFixCallOnMachineFunction(*MaybeMF);
    Changed |= NewChanged;
  }

  for (Function &F : M.functions()) {
    MachineFunction *MaybeMF = MMI.getMachineFunction(F);
    if (!MaybeMF)
      continue;
    this->runVerifyCallOnMachineFunction(*MaybeMF);
  }

  return Changed;
}

static bool findInstrumentNOP(MachineInstr &MI, MachineInstr::MIFlag Flag) {
  int CurOp = 0;
  if (MI.getFlag(Flag))
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isMetadata()) {
        auto *MDN = MO.getMetadata();
        for (const MDOperand &MDO : MDN->operands()) {
          if (auto *MDS = dyn_cast_or_null<MDString>(MDO.get())) {
            if (MDS->getString().equals("JumpCallStackInlineCallStack")) {
              MI.RemoveOperand(CurOp);
              return true;
            }
          }
        }
      }
      CurOp++;
    }
  return false;
}

static MachineBasicBlock::iterator
findInstrumentNOP(MachineBasicBlock &MBB, MachineInstr::MIFlag Flag) {
  for (auto MBBI = MBB.begin(); MBBI != MBB.end(); MBBI++) {
    if (findInstrumentNOP(*MBBI, Flag)) {
      return MBBI;
    }
  }
  return MBB.end();
}

// Fixup prologue and epilogue according to CallStackMethod
bool RISCVJumpCallStack::runFixLabelOnMachineFunction(MachineFunction &MF,
                                                      CallStackMethod CSM) {
  TII = static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  Function &F = MF.getFunction();
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();
  Register SPReg = RISCV::X2;
  Register TReg = RISCV::X7;
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  auto MFI = MF.begin();
  for (; MFI != MF.end(); MFI++) {
    auto &MBB = *MFI;
    MachineBasicBlock::iterator MBBI =
        findInstrumentNOP(MBB, MachineInstr::MIFlag::FrameSetup);
    if (MBBI == MBB.end())
      continue;
    const DebugLoc &DL = MBBI->getDebugLoc();
    switch (CSM) {
    default:
      report_fatal_error("RISCVJumpCallStack: Unsupported CallStackMethod");
      break;
    case JCS_Inline: {
      MachineOptimizationRemark R(DEBUG_TYPE, "InlineCallStackSaved", DL, &MBB);
      R << "Inlined RA Save in " << MF.getName();
      MORE.emit(R);
      const auto &STI = MF.getSubtarget<RISCVSubtarget>();
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI), TReg)
          .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                             RISCVII::MO_HI);
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD))
          .addReg(TReg)
          .addReg(TReg)
          .addReg(SPReg);
      BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
          .addReg(STI.getRegisterInfo()->getRARegister())
          .addReg(TReg)
          .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                             RISCVII::MO_LO);
      break;
    }
    case JCS_Jump:
      if (!(F.hasExternalLinkage() || F.hasAddressTaken() ||
            F.isWeakForLinker())) {
        MachineOptimizationRemark R(DEBUG_TYPE, "JumpCallStackRenamed", DL,
                                    &MBB);
        R << "Renamed " << MF.getName() << " because all callers use jumpoline";
        MORE.emit(R);
        F.setName(getNewDestinationName(MF.getName()));
      } else {
        // Store return address to jump call stack
        // call t0, __tcb_jumpoline_pop
        BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoCALLReg))
            .addReg(RISCV::X5)
            .addExternalSymbol(ClJumpCallStackJumpolinePop.c_str(),
                               RISCVII::MO_CALL)
            .setMIFlag(MachineInstr::FrameSetup);
        MCSymbol *Label = MF.getContext().getOrCreateSymbol(
            getNewDestinationName(MF.getName()));
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
      break;
    }
    MBBI->eraseFromParent();
    Modified = true;
    break;
  }
  for (; MFI != MF.end(); MFI++) {
    auto &MBB = *MFI;
    MachineBasicBlock::iterator MBBI =
        findInstrumentNOP(MBB, MachineInstr::MIFlag::FrameDestroy);
    if (MBBI == MBB.end())
      continue;
    const DebugLoc &DL = MBBI->getDebugLoc();
    // // Load return address from jump call stack
    // // lui     t2, %hi(__tcb_sp_offset)
    // // add     t2, t2, sp
    // // l[w|d]  ra, %lo(__tcb_sp_offset)(t2)
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI))
        .addReg(TReg, RegState::Define)
        .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                           RISCVII::MO_HI)
        .setMIFlag(MachineInstr::FrameDestroy);
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD))
        .addReg(TReg)
        .addReg(TReg)
        .addReg(SPReg)
        .setMIFlag(MachineInstr::FrameDestroy);
    BuildMI(MBB, MBBI, DL, TII->get(IsRV64 ? RISCV::LD : RISCV::LW))
        .addReg(RAReg)
        .addReg(TReg)
        .addExternalSymbol(ClJumpInlineCallStackPointerOffset.c_str(),
                           RISCVII::MO_LO)
        .setMIFlag(MachineInstr::FrameDestroy);
    MBBI->eraseFromParent();
    Modified = true;
    break;
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
    return expandCall(MBB, MBBI, NextMBBI, MORE);
  }
  return false;
}

bool RISCVJumpCallStack::expandCall(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    MachineBasicBlock::iterator &NextMBBI,
                                    MachineOptimizationRemarkEmitter &MORE) {
  static const unsigned OperandNo = 0;
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  MachineOperand &MO = MI.getOperand(OperandNo);
  DebugLoc DL = MI.getDebugLoc();
  StringRef OrigName = getOperandName(MO);
  MCContext &MC = MF->getContext();
  if (OrigName.empty()) {
    MachineOptimizationRemarkMissed R(DEBUG_TYPE, "NotFixedEmpty", DL, &MBB);
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
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "NotFixedNoDest", DL, &MBB);
      R << "Not fixing [no DestSym=" << NewName << "] from " << OrigName;
      MORE.emit(R);
      // no rewrite symbol; just continue
      return false;
    }
    if (auto *ELFLabel = dyn_cast<MCSymbolELF>(DestSym)) {
      if (ELFLabel->getBinding() == ELF::STB_WEAK) {
        MachineOptimizationRemarkMissed R(DEBUG_TYPE, "NotFixedDestWeak", DL,
                                          &MBB);
        R << "Ignoring rename for weak binding of " << NewName << "\n";
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

  // Tell AsmPrinter that we unconditionally want the symbol of this label to
  // be emitted.
  NewMBB->setLabelMustBeEmitted();

  MF->insert(++MBB.getIterator(), NewMBB);

  auto FirstMI = BuildMI(NewMBB, DL, TII->get(RISCV::AUIPC), RISCV::X6);

  if (DestSym) {
    // Ensure that DestSym has a null-terminated name
    assert(strnlen(DestSym->getName().data(), NewName.length() + 1) ==
           NewName.length());

    FirstMI.addExternalSymbol(DestSym->getName().data(), RISCVII::MO_PCREL_HI);
    MachineOptimizationRemark R(DEBUG_TYPE, "FixedLabel", DL, &MBB);
    R << "Fixing reference from " << OrigName << " to " << NewName << "\n";
    MORE.emit(R);
  } else {
    // Copy old operand
    FirstMI.addGlobalAddress(MO.getGlobal(), 0, RISCVII::MO_PCREL_HI);
    MachineOptimizationRemark R(DEBUG_TYPE, "FixedValue", DL, &MBB);
    R << "Fixing reference for " << OrigName << "\n";
    MORE.emit(R);
  }
  BuildMI(NewMBB, DL, TII->get(RISCV::ADDI), RISCV::X6)
      .addReg(RISCV::X6)
      .addMBB(NewMBB, RISCVII::MO_PCREL_LO);
  BuildMI(NewMBB, DL, TII->get(RISCV::PseudoCALLReg), RISCV::X1 /* ra */)
      .addReg(RISCV::X6, RegState::ImplicitKill)
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

void RISCVJumpCallStack::runVerifyCallOnMachineFunction(MachineFunction &MF) {
  MachineOptimizationRemarkEmitter MORE(MF, nullptr);
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case RISCV::PseudoCALL:
        auto OpName = getOperandName(MI.getOperand(0));
        if (OpName.endswith(ClJumpCallStackPostfix) || OpName.empty()) {
          MachineOptimizationRemarkMissed R(DEBUG_TYPE, "FailedCallRename",
                                            MI.getDebugLoc(), MI.getParent());
          MORE.emit(R);
        }
      }
    }
  }
}

/// Returns an instance of the Merge Base Offset Optimization pass.
ModulePass *llvm::createRISCVJumpCallStackPass() {
  return new RISCVJumpCallStack();
}
