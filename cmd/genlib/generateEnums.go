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
	"strings"

	"github.com/bhojpur/neuron/pkg/bindgen"
	"modernc.org/cc"
)

// genCUresult represents a list of enums we want to generate
var genCUreuslt = map[bindgen.TypeKey]struct{}{
	{Kind: cc.Enum, Name: "CUresult"}: {},
}

var cuResultMappings = map[bindgen.TypeKey]string{
	{Kind: cc.Enum, Name: "CUresult"}: "cuResult",
}

func goRenameCUResult(a string) string {
	a = strings.TrimPrefix(a, "CUDA_")
	a = strings.TrimPrefix(a, "ERROR_")
	splits := strings.Split(a, "_")
	for i, s := range splits {
		splits[i] = strings.Title(strings.ToLower(s))
	}
	return strings.Join(splits, "")
}

func generateResultEnums(f io.Writer) {
	t, err := bindgen.Parse(bindgen.Model(), "cuda.h")
	if err != nil {
		panic(err)
	}

	enums := func(decl *cc.Declarator) bool {
		name := bindgen.NameOf(decl)
		kind := decl.Type.Kind()
		tk := bindgen.TypeKey{Kind: kind, Name: name}
		if _, ok := genCUreuslt[tk]; ok {
			return true
		}
		return false
	}
	decls, err := bindgen.Get(t, enums)
	if err != nil {
		panic(err)
	}

	var m []string
	for _, d := range decls {
		e := d.(*bindgen.Enum)
		tk := bindgen.TypeKey{Kind: cc.Enum, Name: e.Name}
		fmt.Fprintf(f, "type %v int\nconst (\n", cuResultMappings[tk])

		// then write the const definitions:
		// 	const(...)

		for _, a := range e.Type.EnumeratorList() {
			enumName := string(a.DefTok.S())
			goName := goRenameCUResult(enumName)
			m = append(m, goName)
			fmt.Fprintf(f, "%v %v = C.%v\n", goName, cuResultMappings[tk], enumName)
		}
		f.Write([]byte(")\n"))
	}
	fmt.Fprintf(f, "var resString = map[cuResult]string{\n")
	for _, s := range m {
		fmt.Fprintf(f, "%v: %q,\n", s, s)
	}
	f.Write([]byte("}\n"))

}
