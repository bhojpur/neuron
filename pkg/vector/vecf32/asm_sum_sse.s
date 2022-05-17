//+build sse
//+build amd64
//+build !noasm !appengine


// AUTO-GENERATED BY C2GOASM -- DO NOT EDIT

#include "textflag.h"
TEXT ·sum(SB), NOSPLIT, $0

	MOVQ a+0(FP), DI
	MOVQ l+8(FP), SI
	MOVQ retVal+24(FP), DX

	WORD $0xf685             // test    esi, esi
	JLE  LBB0_1
	WORD $0x8941; BYTE $0xf0 // mov    r8d, esi
	WORD $0xfe83; BYTE $0x07 // cmp    esi, 7
	JA   LBB0_4
	WORD $0x570f; BYTE $0xc9 // xorps    xmm1, xmm1
	WORD $0xc931             // xor    ecx, ecx
	JMP  LBB0_12

LBB0_1:
	WORD $0x570f; BYTE $0xc9 // xorps    xmm1, xmm1
	JMP  LBB0_13

LBB0_4:
	WORD $0x8944; BYTE $0xc1 // mov    ecx, r8d
	WORD $0xe183; BYTE $0xf8 // and    ecx, -8
	LONG $0xf8718d48         // lea    rsi, [rcx - 8]
	WORD $0x8948; BYTE $0xf0 // mov    rax, rsi
	LONG $0x03e8c148         // shr    rax, 3
	WORD $0xff48; BYTE $0xc0 // inc    rax
	WORD $0x8941; BYTE $0xc1 // mov    r9d, eax
	LONG $0x03e18341         // and    r9d, 3
	LONG $0x18fe8348         // cmp    rsi, 24
	JAE  LBB0_6
	WORD $0x570f; BYTE $0xc0 // xorps    xmm0, xmm0
	WORD $0xc031             // xor    eax, eax
	WORD $0x570f; BYTE $0xc9 // xorps    xmm1, xmm1
	WORD $0x854d; BYTE $0xc9 // test    r9, r9
	JNE  LBB0_9
	JMP  LBB0_11

LBB0_6:
	LONG $0x000001be; BYTE $0x00 // mov    esi, 1
	WORD $0x2948; BYTE $0xc6     // sub    rsi, rax
	LONG $0x31748d49; BYTE $0xff // lea    rsi, [r9 + rsi - 1]
	WORD $0x570f; BYTE $0xc0     // xorps    xmm0, xmm0
	WORD $0xc031                 // xor    eax, eax
	WORD $0x570f; BYTE $0xc9     // xorps    xmm1, xmm1

LBB0_7:
	LONG $0x8714100f             // movups    xmm2, oword [rdi + 4*rax]
	WORD $0x580f; BYTE $0xd0     // addps    xmm2, xmm0
	LONG $0x8744100f; BYTE $0x10 // movups    xmm0, oword [rdi + 4*rax + 16]
	WORD $0x580f; BYTE $0xc1     // addps    xmm0, xmm1
	LONG $0x874c100f; BYTE $0x20 // movups    xmm1, oword [rdi + 4*rax + 32]
	LONG $0x875c100f; BYTE $0x30 // movups    xmm3, oword [rdi + 4*rax + 48]
	LONG $0x8764100f; BYTE $0x40 // movups    xmm4, oword [rdi + 4*rax + 64]
	WORD $0x580f; BYTE $0xe1     // addps    xmm4, xmm1
	WORD $0x580f; BYTE $0xe2     // addps    xmm4, xmm2
	LONG $0x8754100f; BYTE $0x50 // movups    xmm2, oword [rdi + 4*rax + 80]
	WORD $0x580f; BYTE $0xd3     // addps    xmm2, xmm3
	WORD $0x580f; BYTE $0xd0     // addps    xmm2, xmm0
	LONG $0x8744100f; BYTE $0x60 // movups    xmm0, oword [rdi + 4*rax + 96]
	WORD $0x580f; BYTE $0xc4     // addps    xmm0, xmm4
	LONG $0x874c100f; BYTE $0x70 // movups    xmm1, oword [rdi + 4*rax + 112]
	WORD $0x580f; BYTE $0xca     // addps    xmm1, xmm2
	LONG $0x20c08348             // add    rax, 32
	LONG $0x04c68348             // add    rsi, 4
	JNE  LBB0_7
	WORD $0x854d; BYTE $0xc9     // test    r9, r9
	JE   LBB0_11

LBB0_9:
	LONG $0x87448d48; BYTE $0x10 // lea    rax, [rdi + 4*rax + 16]
	WORD $0xf749; BYTE $0xd9     // neg    r9

LBB0_10:
	LONG $0xf050100f         // movups    xmm2, oword [rax - 16]
	WORD $0x580f; BYTE $0xc2 // addps    xmm0, xmm2
	WORD $0x100f; BYTE $0x10 // movups    xmm2, oword [rax]
	WORD $0x580f; BYTE $0xca // addps    xmm1, xmm2
	LONG $0x20c08348         // add    rax, 32
	WORD $0xff49; BYTE $0xc1 // inc    r9
	JNE  LBB0_10

LBB0_11:
	WORD $0x580f; BYTE $0xc1 // addps    xmm0, xmm1
	WORD $0x280f; BYTE $0xc8 // movaps    xmm1, xmm0
	WORD $0x120f; BYTE $0xc8 // movhlps    xmm1, xmm0
	WORD $0x580f; BYTE $0xc8 // addps    xmm1, xmm0
	LONG $0xc97c0ff2         // haddps    xmm1, xmm1
	WORD $0x394c; BYTE $0xc1 // cmp    rcx, r8
	JE   LBB0_13

LBB0_12:
	LONG $0x0c580ff3; BYTE $0x8f // addss    xmm1, dword [rdi + 4*rcx]
	WORD $0xff48; BYTE $0xc1     // inc    rcx
	WORD $0x3949; BYTE $0xc8     // cmp    r8, rcx
	JNE  LBB0_12

LBB0_13:
	LONG $0x0a110ff3 // movss    dword [rdx], xmm1
	RET