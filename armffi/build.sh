clang arm_asm.c -c -Wall -o lib/libarmffi.o
/opt/homebrew/opt/llvm/bin/llvm-ar rc lib/libarmffi.a lib/libarmffi.o
