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
	"io"
	"reflect"
	"text/template"
)

type EngineArith struct {
	Name           string
	VecVar         string
	PrepData       string
	TypeClassCheck string
	IsCommutative  bool

	VV      bool
	LeftVec bool
}

func (fn *EngineArith) methName() string {
	switch {
	case fn.VV:
		return fn.Name
	default:
		return fn.Name + "Scalar"
	}
}

func (fn *EngineArith) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.VV:
		paramNames = []string{"a", "b", "opts"}
		paramTemplates = []*template.Template{tensorType, tensorType, splatFuncOptType}
	default:
		paramNames = []string{"t", "s", "leftTensor", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, boolType, splatFuncOptType}
	}
	return &Signature{
		Name:           fn.methName(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            false,
	}
}

func (fn *EngineArith) WriteBody(w io.Writer) {
	var prep *template.Template
	switch {
	case fn.VV:
		prep = prepVV
		fn.VecVar = "a"
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "t"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "t"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	template.Must(prep.New("prep").Parse(arithPrepRaw))
	prep.Execute(w, fn)
	agg2Body.Execute(w, fn)
}

func (fn *EngineArith) Write(w io.Writer) {
	if tmpl, ok := arithDocStrings[fn.methName()]; ok {
		type tmp struct {
			Left, Right string
		}
		var ds tmp
		if fn.VV {
			ds.Left = "a"
			ds.Right = "b"
		} else {
			ds.Left = "t"
			ds.Right = "s"
		}
		tmpl.Execute(w, ds)
	}

	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("(retVal Tensor, err error) {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func GenerateStdEngArith(f io.Writer, ak Kinds) {
	importStmt := `
import (
	"github.com/pkg/errors"

	"github.com/bhojpur/neuron/pkg/tensor/internal/storage"
)
`
	f.Write([]byte(importStmt))
	var methods []*EngineArith
	for _, abo := range arithBinOps {
		meth := &EngineArith{
			Name:           abo.Name(),
			VV:             true,
			TypeClassCheck: "Number",
			IsCommutative:  abo.IsCommutative,
		}
		methods = append(methods, meth)
	}

	// VV
	for _, meth := range methods {
		meth.Write(f)
		meth.VV = false
	}

	// Scalar
	for _, meth := range methods {
		meth.Write(f)
		meth.LeftVec = true
	}

}

type EngineCmp struct {
	Name           string
	VecVar         string
	PrepData       string
	TypeClassCheck string
	Inv            string

	VV      bool
	LeftVec bool
}

func (fn *EngineCmp) methName() string {
	switch {
	case fn.VV:
		if fn.Name == "Eq" || fn.Name == "Ne" {
			return "El" + fn.Name
		}
		return fn.Name
	default:
		return fn.Name + "Scalar"
	}
}

func (fn *EngineCmp) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.VV:
		paramNames = []string{"a", "b", "opts"}
		paramTemplates = []*template.Template{tensorType, tensorType, splatFuncOptType}
	default:
		paramNames = []string{"t", "s", "leftTensor", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, boolType, splatFuncOptType}
	}
	return &Signature{
		Name:           fn.methName(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            false,
	}
}

func (fn *EngineCmp) WriteBody(w io.Writer) {
	var prep *template.Template
	switch {
	case fn.VV:
		prep = prepVV
		fn.VecVar = "a"
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "t"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "t"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	template.Must(prep.New("prep").Parse(cmpPrepRaw))
	prep.Execute(w, fn)
	agg2CmpBody.Execute(w, fn)
}

func (fn *EngineCmp) Write(w io.Writer) {
	if tmpl, ok := cmpDocStrings[fn.methName()]; ok {
		type tmp struct {
			Left, Right string
		}
		var ds tmp
		if fn.VV {
			ds.Left = "a"
			ds.Right = "b"
		} else {
			ds.Left = "t"
			ds.Right = "s"
		}
		tmpl.Execute(w, ds)
	}
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("(retVal Tensor, err error) {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func GenerateStdEngCmp(f io.Writer, ak Kinds) {
	var methods []*EngineCmp

	for _, abo := range cmpBinOps {
		var tc string
		if abo.Name() == "Eq" || abo.Name() == "Ne" {
			tc = "Eq"
		} else {
			tc = "Ord"
		}
		meth := &EngineCmp{
			Name:           abo.Name(),
			Inv:            abo.Inv,
			VV:             true,
			TypeClassCheck: tc,
		}
		methods = append(methods, meth)
	}

	// VV
	for _, meth := range methods {
		meth.Write(f)
		meth.VV = false
	}

	// Scalar
	for _, meth := range methods {
		meth.Write(f)
		meth.LeftVec = true
	}
}

type EngineMinMax struct {
	Name           string
	VecVar         string
	PrepData       string
	TypeClassCheck string
	Kinds          []reflect.Kind

	VV      bool
	LeftVec bool
}

func (fn *EngineMinMax) methName() string {
	switch {
	case fn.VV:
		return fn.Name
	default:
		return fn.Name + "Scalar"
	}
}

func (fn *EngineMinMax) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.VV:
		paramNames = []string{"a", "b", "opts"}
		paramTemplates = []*template.Template{tensorType, tensorType, splatFuncOptType}
	default:
		paramNames = []string{"t", "s", "leftTensor", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, boolType, splatFuncOptType}
	}
	return &Signature{
		Name:           fn.methName(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            false,
	}
}

func (fn *EngineMinMax) WriteBody(w io.Writer) {
	var prep *template.Template
	switch {
	case fn.VV:
		prep = prepVV
		fn.VecVar = "a"
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "t"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "t"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	template.Must(prep.New("prep").Parse(minmaxPrepRaw))
	prep.Execute(w, fn)
	agg2MinMaxBody.Execute(w, fn)
}

func (fn *EngineMinMax) Write(w io.Writer) {
	if tmpl, ok := cmpDocStrings[fn.methName()]; ok {
		type tmp struct {
			Left, Right string
		}
		var ds tmp
		if fn.VV {
			ds.Left = "a"
			ds.Right = "b"
		} else {
			ds.Left = "t"
			ds.Right = "s"
		}
		tmpl.Execute(w, ds)
	}
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("(retVal Tensor, err error) {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func GenerateStdEngMinMax(f io.Writer, ak Kinds) {
	methods := []*EngineMinMax{
		&EngineMinMax{
			Name:           "MinBetween",
			VV:             true,
			TypeClassCheck: "Ord",
		},
		&EngineMinMax{
			Name:           "MaxBetween",
			VV:             true,
			TypeClassCheck: "Ord",
		},
	}
	f.Write([]byte(`var (
	_ MinBetweener = StdEng{}
	_ MaxBetweener = StdEng{}
)
`))
	// VV
	for _, meth := range methods {
		meth.Write(f)
		meth.VV = false
	}

	// Scalar-Vector
	for _, meth := range methods {
		meth.Write(f)
		meth.LeftVec = true
	}
}

/* UNARY METHODS */

type EngineUnary struct {
	Name           string
	TypeClassCheck string
	Kinds          []reflect.Kind
}

func (fn *EngineUnary) Signature() *Signature {
	return &Signature{
		Name:            fn.Name,
		NameTemplate:    plainName,
		ParamNames:      []string{"a", "opts"},
		ParamTemplates:  []*template.Template{tensorType, splatFuncOptType},
		RetVals:         []string{"retVal"},
		RetValTemplates: []*template.Template{tensorType},

		Err: true,
	}
}

func (fn *EngineUnary) WriteBody(w io.Writer) {
	prepUnary.Execute(w, fn)
	agg2UnaryBody.Execute(w, fn)
}

func (fn *EngineUnary) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("\n}\n"))
}

func GenerateStdEngUncondUnary(f io.Writer, ak Kinds) {
	tcc := []string{
		"Number",     // Neg
		"Number",     // Inv
		"Number",     // Square
		"Number",     // Cube
		"FloatCmplx", // Exp
		"FloatCmplx", // Tanhh
		"FloatCmplx", // Log
		"Float",      // Log2
		"FloatCmplx", // Log10
		"FloatCmplx", // Sqrt
		"Float",      // Cbrt
		"Float",      // InvSqrt
	}
	var gen []*EngineUnary
	for i, u := range unconditionalUnaries {
		var ks []reflect.Kind
		for _, k := range ak.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			ks = append(ks, k)
		}
		fn := &EngineUnary{
			Name:           u.Name(),
			TypeClassCheck: tcc[i],
			Kinds:          ks,
		}
		gen = append(gen, fn)
	}

	for _, fn := range gen {
		fn.Write(f)
	}
}

func GenerateStdEngCondUnary(f io.Writer, ak Kinds) {
	tcc := []string{
		"Signed", // Abs
		"Signed", // Sign
	}
	var gen []*EngineUnary
	for i, u := range conditionalUnaries {
		var ks []reflect.Kind
		for _, k := range ak.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			ks = append(ks, k)
		}
		fn := &EngineUnary{
			Name:           u.Name(),
			TypeClassCheck: tcc[i],
			Kinds:          ks,
		}
		gen = append(gen, fn)
	}

	for _, fn := range gen {
		fn.Write(f)
	}
}
