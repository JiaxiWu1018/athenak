# Session009 technical preparation

Independent source checkout onproject/GI-in-cluster based on pushedc3873b4e0f46ab9397c2f270cb7139515013a0f3; productionrestart-headerrepair263dcf21 preserved byancestry. Kokkos6739bc623081648af9e752b616d9671527922cbf. GWtimestamp precisionfix fe68d684 changes diagnostics only. Physicalsource/pusher unchanged; fresh initialization reads explicit boostinput.

Pinned AMD build stack/toolchain/launcher recorded by scripts/amd_build.sbatch andscripts/amd_env.inc. TwoMPI/HIP binaries: customGI andexistingbuilt-inlinearGWgenerator. Exact finalcommit, input/scripts/executablehashes and jobs are recorded in control/config.json and evidence/SUBMISSION.md. Input/output names androots reviewed and scoped toSession009.

PerseusQA11050 passed9meaningful invariants/horizonjoinchecks andsyntaxchecks; later changes get final allocated verification. ActualAMDmesh/MPI/GPUinitialization/restart/propagation measurements are pending. Stopped numericalfailures retainlogs/checkpoints/failureevidence, neverretryautomatically. Source/code/, inputs/, scripts/, docs/evidence retained; no fullrawvolume/checkpoint copied toPerseus. Exactcommands inREADME.

Operations implement one-successor Slurmcontinuation, full-wallallocationbudget reservation≤10000nodeh, ≤90segments/45days, stickyUSER_STOP, source/config bindings, signed archivehashreceipts andlatestthree checkpointretention. Antacronmetadataonly; alltransfer/reduction/render onallocatednodes. Sealedmanifestincludes checkpoints. Single sphere complex77multipoles preserved; comparison/FFI/gaps honest. Initial t12stage is productionhistory, notdiscardedpilot.
