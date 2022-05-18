package main

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

import (
	"fmt"
	"io"
)

func filterCSigs(sigs []*CSignature) (retVal []*CSignature) {
	for _, sig := range sigs {
		if _, ok := ignoredFunctions[sig.Name]; ok {
			continue
		}
		retVal = append(retVal, sig)
	}
	return
}

func cgoCall(buf io.Writer, sig *CSignature) {
	fmt.Fprintf(buf, "result(C.%s(", sig.Name)
	for i, param := range sig.Params {
		if param.Type == "void" && !param.IsPtr {
			continue
		}
		if param.IsPtr && param.IsRetVal {
			fmt.Fprintf(buf, "&C%s", param.Name)
		} else {
			fmt.Fprintf(buf, "C%s", param.Name)
		}

		if i < len(sig.Params)-1 {
			buf.Write([]byte(", "))
		}
	}
	fmt.Fprintf(buf, "))\n")
}

func go2CParam(buf io.Writer, goParam, cParam *Param) {
	if cParam == nil {
		panic("WTF?")
	}
	ctype, ok := gotypesConversion[goParam.Type]
	if !ok {
		panic(fmt.Sprintf("Go type %q does not have conversion to C type", goParam.Type))
	}
	conv := fmt.Sprintf(ctype, goParam.Name)
	fmt.Fprintf(buf, "C%s := %s\n", cParam.Name, conv)
}
