//+build sse
//+build amd64
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

// But clearlly edited by me to give it some customizations
#include "textflag.h"
TEXT ·sum(SB), NOSPLIT, $0

	MOVQ a+0(FP), DI
	MOVQ l+8(FP), SI
	MOVQ retVal+24(FP), DX

	WORD $0xf685             // test    esi, esi
	JLE  LBB0_1
	WORD $0x8941; BYTE $0xf0 // mov    r8d, esi
	WORD $0xfe83; BYTE $0x03 // cmp    esi, 3
	JA   LBB0_4
	LONG $0xc0570f66         // xorpd    xmm0, xmm0
	WORD $0xc931             // xor    ecx, ecx
	JMP  LBB0_12

LBB0_1:
	LONG $0xc0570f66 // xorpd    xmm0, xmm0
	JMP  LBB0_13

LBB0_4:
	WORD $0x8944; BYTE $0xc1 // mov    ecx, r8d
	WORD $0xe183; BYTE $0xfc // and    ecx, -4
	LONG $0xfc718d48         // lea    rsi, [rcx - 4]
	WORD $0x8948; BYTE $0xf0 // mov    rax, rsi
	LONG $0x02e8c148         // shr    rax, 2
	WORD $0xff48; BYTE $0xc0 // inc    rax
	WORD $0x8941; BYTE $0xc1 // mov    r9d, eax
	LONG $0x03e18341         // and    r9d, 3
	LONG $0x0cfe8348         // cmp    rsi, 12
	JAE  LBB0_6
	LONG $0xc0570f66         // xorpd    xmm0, xmm0
	WORD $0xc031             // xor    eax, eax
	LONG $0xc9570f66         // xorpd    xmm1, xmm1
	WORD $0x854d; BYTE $0xc9 // test    r9, r9
	JNE  LBB0_9
	JMP  LBB0_11

LBB0_6:
	LONG $0x000001be; BYTE $0x00 // mov    esi, 1
	WORD $0x2948; BYTE $0xc6     // sub    rsi, rax
	LONG $0x31748d49; BYTE $0xff // lea    rsi, [r9 + rsi - 1]
	LONG $0xc0570f66             // xorpd    xmm0, xmm0
	WORD $0xc031                 // xor    eax, eax
	LONG $0xc9570f66             // xorpd    xmm1, xmm1

LBB0_7:
	LONG $0x14100f66; BYTE $0xc7   // movupd    xmm2, oword [rdi + 8*rax]
	LONG $0xd0580f66               // addpd    xmm2, xmm0
	LONG $0x44100f66; WORD $0x10c7 // movupd    xmm0, oword [rdi + 8*rax + 16]
	LONG $0xc1580f66               // addpd    xmm0, xmm1
	LONG $0x4c100f66; WORD $0x20c7 // movupd    xmm1, oword [rdi + 8*rax + 32]
	LONG $0x5c100f66; WORD $0x30c7 // movupd    xmm3, oword [rdi + 8*rax + 48]
	LONG $0x64100f66; WORD $0x40c7 // movupd    xmm4, oword [rdi + 8*rax + 64]
	LONG $0xe1580f66               // addpd    xmm4, xmm1
	LONG $0xe2580f66               // addpd    xmm4, xmm2
	LONG $0x54100f66; WORD $0x50c7 // movupd    xmm2, oword [rdi + 8*rax + 80]
	LONG $0xd3580f66               // addpd    xmm2, xmm3
	LONG $0xd0580f66               // addpd    xmm2, xmm0
	LONG $0x44100f66; WORD $0x60c7 // movupd    xmm0, oword [rdi + 8*rax + 96]
	LONG $0xc4580f66               // addpd    xmm0, xmm4
	LONG $0x4c100f66; WORD $0x70c7 // movupd    xmm1, oword [rdi + 8*rax + 112]
	LONG $0xca580f66               // addpd    xmm1, xmm2
	LONG $0x10c08348               // add    rax, 16
	LONG $0x04c68348               // add    rsi, 4
	JNE  LBB0_7
	WORD $0x854d; BYTE $0xc9       // test    r9, r9
	JE   LBB0_11

LBB0_9:
	LONG $0xc7448d48; BYTE $0x10 // lea    rax, [rdi + 8*rax + 16]
	WORD $0xf749; BYTE $0xd9     // neg    r9

LBB0_10:
	LONG $0x50100f66; BYTE $0xf0 // movupd    xmm2, oword [rax - 16]
	LONG $0xc2580f66             // addpd    xmm0, xmm2
	LONG $0x10100f66             // movupd    xmm2, oword [rax]
	LONG $0xca580f66             // addpd    xmm1, xmm2
	LONG $0x20c08348             // add    rax, 32
	WORD $0xff49; BYTE $0xc1     // inc    r9
	JNE  LBB0_10

LBB0_11:
	LONG $0xc1580f66         // addpd    xmm0, xmm1
	LONG $0xc07c0f66         // haddpd    xmm0, xmm0
	WORD $0x394c; BYTE $0xc1 // cmp    rcx, r8
	JE   LBB0_13

LBB0_12:
	LONG $0x04580ff2; BYTE $0xcf // addsd    xmm0, qword [rdi + 8*rcx]
	WORD $0xff48; BYTE $0xc1     // inc    rcx
	WORD $0x3949; BYTE $0xc8     // cmp    r8, rcx
	JNE  LBB0_12

LBB0_13:
	LONG $0x02110ff2 // movsd    qword [rdx], xmm0
	RET