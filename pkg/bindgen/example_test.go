package bindgen_test

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
	"strings"

	"github.com/bhojpur/neuron/pkg/bindgen"
	"modernc.org/cc"
)

// GoSignature represents a Go signature
type GoSignature struct {
	Receiver Param
	Name     string
	Params   []Param
	Ret      []Param
}

func (sig *GoSignature) Format(f fmt.State, c rune) {
	f.Write([]byte("func "))
	if sig.Receiver.Name != "" {
		fmt.Fprintf(f, "(%v %v) ", sig.Receiver.Name, sig.Receiver.Type)
	}
	fmt.Fprintf(f, "%v(", sig.Name)
	for i, p := range sig.Params {
		fmt.Fprintf(f, "%v %v", p.Name, p.Type)
		if i < len(sig.Params)-1 {
			fmt.Fprint(f, ", ")
		}
	}
	fmt.Fprint(f, ")")

	switch len(sig.Ret) {
	case 0:
		return
	default:
		fmt.Fprint(f, " (")
		for i, p := range sig.Ret {
			fmt.Fprintf(f, "%v %v", p.Name, p.Type)
			if i < len(sig.Ret)-1 {
				fmt.Fprint(f, ", ")
			}
		}
		fmt.Fprint(f, ")")
	}
}

// Param represents the parameters in Go
type Param struct {
	Name, Type string
}

// functions say we only want functions declared
func functions(t *cc.TranslationUnit) ([]bindgen.Declaration, error) {
	filter := func(d *cc.Declarator) bool {
		if !strings.HasPrefix(bindgen.NameOf(d), "func") {
			return false
		}
		if d.Type.Kind() != cc.Function {
			return false
		}
		return true
	}
	return bindgen.Get(t, filter)
}

func decl2GoSig(d *bindgen.CSignature) *GoSignature {
	var params []Param
	sig := new(GoSignature)
outer:
	for _, p := range d.Parameters() {
		// check if its a receiver
		if ctxrec, ok := contextualFns[d.Name]; ok {
			if ctxrec.Name == p.Name() {
				sig.Receiver = ctxrec
				continue
			}
		}
		Type := cleanType(p.Type())
		if retP, ok := retVals[d.Name]; ok {
			for _, r := range retP {
				if p.Name() == r {
					sig.Ret = append(sig.Ret, Param{p.Name(), Type})
					continue outer
				}
			}
		}
		params = append(params, Param{p.Name(), Type})
	}
	retType := cleanType(d.Return)
	if !bindgen.IsVoid(d.Return) {
		sig.Ret = append(sig.Ret, Param{"err", retType})
	}

	sig.Name = d.Name
	sig.Params = params
	return sig
}

func cleanType(t cc.Type) string {
	Type := t.String()
	if td := bindgen.TypeDefOf(t); td != "" {
		Type = td
	}

	if bindgen.IsConstType(t) {
		Type = strings.TrimPrefix(Type, "const ")
	}
	if bindgen.IsPointer(t) {
		Type = strings.TrimSuffix(Type, "*")
	}
	return Type
}

var contextualFns = map[string]Param{
	"funcCtx": Param{"ctx", "Ctx"},
}

var retVals = map[string][]string{
	"funcErr": []string{"retVal"},
	"funcCtx": []string{"retVal"},
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func Example_simple() {
	t, err := bindgen.Parse(bindgen.Model(), "testdata/dummy.h")
	handleErr(err)
	fns, err := functions(t)
	handleErr(err)
	for _, fn := range fns {
		fmt.Println(decl2GoSig(fn.(*bindgen.CSignature)))
	}

	// Output:
	// func func1i(a int)
	// func func1f(a foo)
	// func func1fp(a foo)
	// func func2i(a int, b int)
	// func func2f(a foo, b int)
	// func funcErr(a int) (retVal foo, err error)
	// func (ctx Ctx) funcCtx(a foo) (retVal foo, err error)
}
