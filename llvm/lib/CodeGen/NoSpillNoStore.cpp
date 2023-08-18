//===-- llvm/CodeGen/NoSpillNoStore.cpp ---------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass verifies that instructions which are marked NoSpill do not have
/// results that are then stored [spilled].
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"
#include <map>
#include <set>
#include <string>
using namespace llvm;

#define DEBUG_TYPE "nospill-nostore"

namespace {

enum NSNSAction { NONE, WARNING, ERROR, FIX };

static cl::opt<NSNSAction>
    ClNSNSAction("nospill-nostore",
                 cl::desc("Action to take in nospill-nostore pass"), cl::Hidden,
#ifdef NDEBUG
                 cl::init(NONE),
#else
                 cl::init(WARNING),
#endif
                 cl::values(clEnumValN(NONE, "none", "nothing"),
                            clEnumValN(WARNING, "warning", "warning"),
                            clEnumValN(ERROR, "error", "error"),
                            clEnumValN(FIX, "fix", "fix")));

class NoSpillNoStore : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  NoSpillNoStore() : MachineFunctionPass(ID) {}

private:
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char NoSpillNoStore::ID = 0;
char &llvm::NoSpillNoStoreID = NoSpillNoStore::ID;
INITIALIZE_PASS(NoSpillNoStore, DEBUG_TYPE,
                "Verify NoSpill results are not stored", false, false)

static std::string getRegisterName(const MCRegisterInfo *TRI,
                                   llvm::Register Reg) {
  if (Reg.isPhysical())
    return TRI->getName(Reg.asMCReg());
  if (Reg.isVirtual())
    return llvm::formatv("V{0}", Reg.virtRegIndex());
  return llvm::formatv("{0}", Reg.id());
}

struct NoSpillRegisterTracker {
  llvm::Register Register;
  MachineOperand *Defined;
  // MachineInstr *LastUsed;
  MachineOperand *LastUsed;
  NoSpillRegisterTracker(/* llvm::Register Register, */
                         /* MachineInstr *Defined,  */ MachineOperand *LastUsed)
      : Register(LastUsed->getReg()), Defined(LastUsed), LastUsed(LastUsed) {
    assert(LastUsed->getParent() && "LastUsed must have a Machine Instruction");
  }
};

bool NoSpillNoStore::runOnMachineFunction(MachineFunction &MF) {

  if (ClNSNSAction == NONE)
    return false;

  auto *TRI = MF.getTarget().getMCRegisterInfo();
  // auto *TargetRI = MF.getSubtarget().getRegisterInfo();
  // auto *TargetII = MF.getSubtarget().getInstrInfo();
  auto *TII = MF.getTarget().getMCInstrInfo();
  auto *MRI = &MF.getRegInfo();
  bool Changed = false;

  LLVM_DEBUG(errs() << "Running " DEBUG_TYPE " on " << MF.getName() << "\n");

  const Function &F = MF.getFunction();
  const Module *M = F.getParent();
  ModuleSlotTracker MST(M);
  MST.incorporateFunction(F);

  // Iterate through each instruction in the function, looking for pseudos.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    LLVM_DEBUG(errs() << "    ");
    LLVM_DEBUG(MBB->printName(errs(),
                              MachineBasicBlock::PrintNameIr |
                                  MachineBasicBlock::PrintNameAttributes,
                              &MST));
    LLVM_DEBUG(errs() << "\n");
    std::map<llvm::Register, std::unique_ptr<NoSpillRegisterTracker>>
        UsedRegisters;
    std::set<llvm::Register> NeedsReload;
    std::map<llvm::Register, llvm::Register> ReMapped;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE;) {
      MachineBasicBlock::iterator CurMBBI = MBBI;
      MachineInstr &MI = *MBBI++;

      auto const &MII = TII->get(MI.getOpcode());
      bool MayStore = MII.mayStore();
      bool IsCall = MII.isCall();
      bool IsNoSpill = MI.getNoSpill();

      std::set<llvm::Register> Marked;
      std::set<llvm::Register> Killed;
      for (auto &Operand : MI.operands()) {
        if (!Operand.isReg())
          continue;
        auto R = Operand.getReg();
        if (MayStore) {
          auto It = UsedRegisters.find(R);
          if (It != UsedRegisters.end()) {
            errs() << "Storing NoSpill Value " << getRegisterName(TRI, R)
                   << " at ";
            // MI.getDebugLoc().print(errs());
            // errs() << "\n";
            MI.print(errs());
            if (ClNSNSAction == ERROR)
              report_fatal_error("Stored a NoSpill Value");
          }
        }
        if (IsNoSpill && Operand.isDef() && !Operand.isKill()) {
          Marked.insert(R);
          UsedRegisters.insert(std::make_pair(
              R, std::make_unique<NoSpillRegisterTracker>(&Operand)));
        }
        if (!Operand.isDef()) {
          auto URIt = UsedRegisters.find(R);
          if (URIt != UsedRegisters.end()) {
            auto &NSRT = URIt->second;
            auto NRIt = NeedsReload.find(R);
            if (NRIt != NeedsReload.end()) {
              if (ClNSNSAction != FIX)
                report_fatal_error(
                    "Fixing nospill register without FIX action");
              auto *OrigMO = NSRT->Defined;
              auto *OrigMI = OrigMO->getParent();
              auto MIB =
                  llvm::BuildMI(MF, MIMetadata(*OrigMI), OrigMI->getDesc());
              Register NewR = R;
              if (R.isVirtual()) {
                NewR = MRI->cloneVirtualRegister(R);
                ReMapped.insert(std::make_pair(R, NewR));
              }
              for (auto &OrigOp : OrigMI->explicit_operands()) {
                if (!OrigOp.isReg()) {
                  MIB.add(MachineOperand(OrigOp));
                  continue;
                }
                MachineOperand NewMO = MachineOperand::CreateReg(
                    OrigOp.getReg(), OrigOp.isDef(), OrigOp.isImplicit(),
                    OrigOp.isKill(), OrigOp.isDead(), OrigOp.isUndef(),
                    OrigOp.isEarlyClobber(), OrigOp.getSubReg(),
                    OrigOp.isDebug(), OrigOp.isInternalRead(),
                    OrigOp.getReg().isPhysical() && OrigOp.isRenamable());
                if (NewR != R && NewMO.getReg() == R) {
                  NewMO.setReg(NewR);
                  assert(NewMO.isDef() && !NewMO.isKill() &&
                         "Defined and Dead for UsedRegister!!!");
                }
                MIB.add(NewMO);
                if (OrigOp.getReg() == R) {
                  __attribute__((unused)) auto P =
                      UsedRegisters.insert(std::make_pair(
                          NewR,
                          std::make_unique<NoSpillRegisterTracker>(
                              &MIB->getOperand(MIB->getNumOperands() - 1))));
                  assert(P.first->second->Defined->getParent() &&
                         "UsedRegisters grabbed operand without parent");
                }
                // auto OrigR = OrigOp.getReg();
                // MIB.add(MachineOperand::CreateReg(
                //     OrigR == R ? NewR : OrigR, OrigOp.isDef(),
                //     OrigOp.isImplicit(), OrigOp.isKill(), OrigOp.isDead(),
                //     OrigOp.isUndef(), OrigOp.isEarlyClobber(),
                //     OrigOp.getSubReg(), OrigOp.isDebug(),
                //     OrigOp.isInternalRead(),
                //     OrigR.isPhysical() && OrigOp.isRenamable()));
                // if (OrigR == R)
                //   for (MachineOperand &MO : MIB->explicit_operands())
                //     if (MO.isReg() && MO.getReg() == NewR)
                //       UsedRegisters.insert(std::make_pair(
                //           NewR,
                //           std::make_unique<NoSpillRegisterTracker>(&MO)));
              }
              if (NSRT->LastUsed->getParent() == NSRT->Defined->getParent()) {
                NSRT->LastUsed->setIsDead();
                NSRT->LastUsed->setIsUndef();
              } else {
                NSRT->LastUsed->setIsKill();
              }
              MBB->insert(CurMBBI, MIB);
              assert(UsedRegisters.find(NewR) != UsedRegisters.end() &&
                     "UsedRegisters does not have renamed register");
              LLVM_DEBUG(errs() << "        Reloaded "
                                << getRegisterName(TRI, R) << " with ";
                         MIB->print(errs()));
              Changed = true;
              Marked.insert(NewR);
              NeedsReload.erase(NRIt);
            }
            auto RMIt = ReMapped.find(R);
            if (RMIt != ReMapped.end()) {
              if (ClNSNSAction != FIX)
                report_fatal_error(
                    "Fixing nospill register without FIX action");
              Operand.setReg(RMIt->second);
              R = RMIt->second;
            }
          }
        }
        if (Operand.isKill()) {
          if (UsedRegisters.erase(R)) {
            NeedsReload.erase(R);
            Killed.insert(R);
          }
          ReMapped.erase(R);
        }
      }

      if (!Killed.empty()) {
        LLVM_DEBUG(errs() << "        Killing");
        LLVM_DEBUG(for (auto R
                        : Killed) errs()
                   << " " << getRegisterName(TRI, R));
        LLVM_DEBUG(errs() << " at "; MI.print(errs()));
      }

      if (!Marked.empty()) {
        LLVM_DEBUG(errs() << "        Marking NoSpill registers");
        LLVM_DEBUG(for (auto R
                        : Marked) errs()
                   << " " << getRegisterName(TRI, R));
        LLVM_DEBUG(errs() << " at "; MI.print(errs()));
      }

      if (IsCall && !UsedRegisters.empty()) {
        switch (ClNSNSAction) {
        case ERROR:
        case WARNING:
          errs() << "        NoSpill Registers across call :";
          for (auto &R : UsedRegisters)
            R.second->Defined->getParent()->print(errs());
          if (ClNSNSAction == ERROR)
            report_fatal_error("NoSpill Register alive across a call");
          break;
        case NONE:
          break;
        case FIX:
          LLVM_DEBUG(errs() << "        Marking NeedsReload registers");
          for (auto &R : UsedRegisters) {
            LLVM_DEBUG(errs() << " " << getRegisterName(TRI, R.first));
            NeedsReload.insert(R.second->Register);
          }
          LLVM_DEBUG(errs() << " at "; MI.print(errs()));
          break;
        }
      }
    }
    if (!UsedRegisters.empty()) {
      LLVM_DEBUG(errs() << "    Abandoned NoSpills:\n");
      LLVM_DEBUG(for (auto &R
                      : UsedRegisters) {
        errs() << " " << getRegisterName(TRI, R.first) << " from ";
        R.second->Defined->print(errs());
      });
    }
  }

  return Changed;
}
