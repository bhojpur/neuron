//+build !noasm !appengine
// AUTO-GENERATED BY C2GOASM -- DO NOT EDIT

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

TEXT ·__sum(SB), $0-24

	MOVQ a+0(FP), DI
	MOVQ l+8(FP), SI
	MOVQ retVal+16(FP), DX

	WORD $0xf685             // test    esi, esi
	JLE  LBB0_1
	WORD $0x8941; BYTE $0xf1 // mov    r9d, esi
	WORD $0xfe83; BYTE $0x1f // cmp    esi, 31
	JA   LBB0_4
	LONG $0xc057f8c5         // vxorps    xmm0, xmm0, xmm0
	WORD $0xc931             // xor    ecx, ecx
	JMP  LBB0_11

LBB0_1:
	LONG $0xc057f8c5 // vxorps    xmm0, xmm0, xmm0
	JMP  LBB0_12

LBB0_4:
	WORD $0x8944; BYTE $0xc9     // mov    ecx, r9d
	WORD $0xe183; BYTE $0xe0     // and    ecx, -32
	LONG $0xe0718d48             // lea    rsi, [rcx - 32]
	WORD $0x8948; BYTE $0xf0     // mov    rax, rsi
	LONG $0x05e8c148             // shr    rax, 5
	WORD $0xff48; BYTE $0xc0     // inc    rax
	WORD $0x8941; BYTE $0xc0     // mov    r8d, eax
	LONG $0x01e08341             // and    r8d, 1
	WORD $0x8548; BYTE $0xf6     // test    rsi, rsi
	JE   LBB0_5
	LONG $0x000001be; BYTE $0x00 // mov    esi, 1
	WORD $0x2948; BYTE $0xc6     // sub    rsi, rax
	LONG $0x30448d49; BYTE $0xff // lea    rax, [r8 + rsi - 1]
	LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
	WORD $0xf631                 // xor    esi, esi
	LONG $0xc957f0c5             // vxorps    xmm1, xmm1, xmm1
	LONG $0xd257e8c5             // vxorps    xmm2, xmm2, xmm2
	LONG $0xdb57e0c5             // vxorps    xmm3, xmm3, xmm3

LBB0_7:
	LONG $0x0458fcc5; BYTE $0xb7         // vaddps    ymm0, ymm0, yword [rdi + 4*rsi]
	LONG $0x4c58f4c5; WORD $0x20b7       // vaddps    ymm1, ymm1, yword [rdi + 4*rsi + 32]
	LONG $0x5458ecc5; WORD $0x40b7       // vaddps    ymm2, ymm2, yword [rdi + 4*rsi + 64]
	LONG $0x5c58e4c5; WORD $0x60b7       // vaddps    ymm3, ymm3, yword [rdi + 4*rsi + 96]
	QUAD $0x000080b78458fcc5; BYTE $0x00 // vaddps    ymm0, ymm0, yword [rdi + 4*rsi + 128]
	QUAD $0x0000a0b78c58f4c5; BYTE $0x00 // vaddps    ymm1, ymm1, yword [rdi + 4*rsi + 160]
	QUAD $0x0000c0b79458ecc5; BYTE $0x00 // vaddps    ymm2, ymm2, yword [rdi + 4*rsi + 192]
	QUAD $0x0000e0b79c58e4c5; BYTE $0x00 // vaddps    ymm3, ymm3, yword [rdi + 4*rsi + 224]
	LONG $0x40c68348                     // add    rsi, 64
	LONG $0x02c08348                     // add    rax, 2
	JNE  LBB0_7
	WORD $0x854d; BYTE $0xc0             // test    r8, r8
	JE   LBB0_10

LBB0_9:
	LONG $0x5c58e4c5; WORD $0x60b7 // vaddps    ymm3, ymm3, yword [rdi + 4*rsi + 96]
	LONG $0x5458ecc5; WORD $0x40b7 // vaddps    ymm2, ymm2, yword [rdi + 4*rsi + 64]
	LONG $0x4c58f4c5; WORD $0x20b7 // vaddps    ymm1, ymm1, yword [rdi + 4*rsi + 32]
	LONG $0x0458fcc5; BYTE $0xb7   // vaddps    ymm0, ymm0, yword [rdi + 4*rsi]

LBB0_10:
	LONG $0xcb58f4c5               // vaddps    ymm1, ymm1, ymm3
	LONG $0xc258fcc5               // vaddps    ymm0, ymm0, ymm2
	LONG $0xc158fcc5               // vaddps    ymm0, ymm0, ymm1
	LONG $0x197de3c4; WORD $0x01c1 // vextractf128    xmm1, ymm0, 1
	LONG $0xc158fcc5               // vaddps    ymm0, ymm0, ymm1
	LONG $0x0579e3c4; WORD $0x01c8 // vpermilpd    xmm1, xmm0, 1
	LONG $0xc158fcc5               // vaddps    ymm0, ymm0, ymm1
	LONG $0xc07cffc5               // vhaddps    ymm0, ymm0, ymm0
	WORD $0x394c; BYTE $0xc9       // cmp    rcx, r9
	JE   LBB0_12

LBB0_11:
	LONG $0x0458fac5; BYTE $0x8f // vaddss    xmm0, xmm0, dword [rdi + 4*rcx]
	WORD $0xff48; BYTE $0xc1     // inc    rcx
	WORD $0x3949; BYTE $0xc9     // cmp    r9, rcx
	JNE  LBB0_11

LBB0_12:
	LONG $0x0211fac5 // vmovss    dword [rdx], xmm0
	VZEROUPPER
	RET

LBB0_5:
	LONG $0xc057f8c5         // vxorps    xmm0, xmm0, xmm0
	WORD $0xf631             // xor    esi, esi
	LONG $0xc957f0c5         // vxorps    xmm1, xmm1, xmm1
	LONG $0xd257e8c5         // vxorps    xmm2, xmm2, xmm2
	LONG $0xdb57e0c5         // vxorps    xmm3, xmm3, xmm3
	WORD $0x854d; BYTE $0xc0 // test    r8, r8
	JNE  LBB0_9
	JMP  LBB0_10
