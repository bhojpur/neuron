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
	"text/template"
)

const unaryTestBodyRaw = `invFn := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}


	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := q.Engine().({{interfaceName .Name}}); we = we || !ok
	
	ret, err := {{.Name}}(a {{template "funcoptuse"}})
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, nil, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{if ne .InvTypeClass "" -}}
	if err := typeclassCheck(a.Dtype(), {{.InvTypeClass}}); err != nil {
		return true // uninvertible due to type class implementation issues
	}
	{{end -}}
	{{if eq .FuncOpt "incr" -}}
	if ret, err = Sub(ret, identityVal(100, a.Dtype()),  UseUnsafe()) ; err != nil {
		t.Errorf("err while subtracting incr: %v", err)
		return false
	}
	{{end -}}
	{{.Inv}}(ret, UseUnsafe())
	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}
	return true
}

if err := quick.Check(invFn, &quick.Config{Rand:newRand(), MaxCount: quickchecks}); err != nil{
	t.Errorf("Inv tests for {{.Name}} failed: %v", err)
}
`

type unaryTest struct {
	unaryOp
	FuncOpt             string
	EqFailTypeClassName string
	InvTypeClass        string
}

func (fn *unaryTest) Name() string {
	if fn.unaryOp.Name() == "Eq" || fn.unaryOp.Name() == "Ne" {
		return "El" + fn.unaryOp.Name()
	}
	return fn.unaryOp.Name()
}

func (fn *unaryTest) Signature() *Signature {
	name := fmt.Sprintf("Test%s", fn.unaryOp.Name())
	if fn.FuncOpt != "" {
		name += "_" + fn.FuncOpt
	}
	return &Signature{
		Name:           name,
		NameTemplate:   plainName,
		ParamNames:     []string{"t"},
		ParamTemplates: []*template.Template{testingType},
	}
}

func (fn *unaryTest) WriteBody(w io.Writer) {
	t := template.Must(template.New("unary test body").Funcs(funcs).Parse(unaryTestBodyRaw))
	template.Must(t.New("funcoptdecl").Parse(funcOptDecl[fn.FuncOpt]))
	template.Must(t.New("funcoptcorrect").Parse(funcOptCorrect[fn.FuncOpt]))
	template.Must(t.New("funcoptuse").Parse(funcOptUse[fn.FuncOpt]))
	template.Must(t.New("funcoptcheck").Parse(funcOptCheck[fn.FuncOpt]))
	t.Execute(w, fn)
}

func (fn *unaryTest) canWrite() bool { return fn.Inv != "" }

func (fn *unaryTest) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func GenerateAPIUnaryTests(f io.Writer, ak Kinds) {
	var tests []*unaryTest
	for _, op := range conditionalUnaries {
		t := &unaryTest{
			unaryOp:             op,
			EqFailTypeClassName: "nil",
		}

		tests = append(tests, t)
	}

	for _, op := range unconditionalUnaries {
		t := &unaryTest{
			unaryOp:             op,
			EqFailTypeClassName: "nil",
		}
		switch op.name {
		case "Square":
			t.InvTypeClass = "floatcmplxTypes"
		case "Cube":
			t.InvTypeClass = "floatTypes"
		}

		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "incr"
	}

	// for now incr cannot be quickchecked

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
	}
}
