package generator

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

type DenseBinOp struct {
	MethodName string
	Name       string
	Scalar     bool
}

func (fn *DenseBinOp) Write(w io.Writer) {
	type tmp struct {
		Left, Right string
	}
	var ds tmp
	ds.Left = "t"
	ds.Right = "other"
	name := fn.MethodName
	if fn.Scalar {
		name += "Scalar"
	}
	if tmpl, ok := arithDocStrings[name]; ok {
		tmpl.Execute(w, ds)
	}
	if tmpl, ok := cmpDocStrings[name]; ok {
		tmpl.Execute(w, ds)
	}

	if fn.Scalar {
		fmt.Fprintf(w, "func (t *Dense) %sScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.MethodName)
		denseArithScalarBody.Execute(w, fn)
	} else {
		fmt.Fprintf(w, "func (t *Dense) %s(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {\n", fn.MethodName)
		denseArithBody.Execute(w, fn)
	}
	w.Write([]byte("}\n\n"))

}

func GenerateDenseArith(f io.Writer, ak Kinds) {
	var methods []*DenseBinOp
	for _, bo := range arithBinOps {
		meth := &DenseBinOp{
			MethodName: bo.Name(),
			Name:       bo.Name(),
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.Scalar = true
	}
	for _, meth := range methods {
		meth.Write(f)
	}
}

func GenerateDenseCmp(f io.Writer, ak Kinds) {
	var methods []*DenseBinOp
	for _, cbo := range cmpBinOps {
		methName := cbo.Name()
		if methName == "Eq" || methName == "Ne" {
			methName = "El" + cbo.Name()
		}
		meth := &DenseBinOp{
			MethodName: methName,
			Name:       cbo.Name(),
		}
		methods = append(methods, meth)
	}
	for _, meth := range methods {
		meth.Write(f)
		meth.Scalar = true
	}
	for _, meth := range methods {
		meth.Write(f)
	}
}
