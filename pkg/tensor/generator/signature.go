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

type Signature struct {
	Name            string
	NameTemplate    *template.Template
	ParamNames      []string
	ParamTemplates  []*template.Template
	RetVals         []string
	RetValTemplates []*template.Template

	Kind reflect.Kind
	Err  bool
}

func (s *Signature) Write(w io.Writer) {
	s.NameTemplate.Execute(w, s)
	w.Write([]byte("("))
	for i, p := range s.ParamTemplates {
		w.Write([]byte(s.ParamNames[i]))
		w.Write([]byte(" "))
		p.Execute(w, s.Kind)

		if i < len(s.ParamNames) {
			w.Write([]byte(", "))
		}
	}
	w.Write([]byte(")"))
	if len(s.RetVals) > 0 {
		w.Write([]byte("("))
		for i, r := range s.RetValTemplates {
			w.Write([]byte(s.RetVals[i]))
			w.Write([]byte(" "))
			r.Execute(w, s.Kind)

			if i < len(s.RetVals) {
				w.Write([]byte(", "))
			}
		}

		if s.Err {
			w.Write([]byte("err error"))
		}
		w.Write([]byte(")"))
		return
	}

	if s.Err {
		w.Write([]byte("(err error)"))
	}
}

const (
	golinkPragmaRaw = "//go:linkname {{.Name}}{{short .Kind}} github.com/bhojpur/neuron/pkg/vector/{{vecPkg .Kind}}{{getalias .Name}}\n"

	typeAnnotatedNameRaw = `{{.Name}}{{short .Kind}}`
	plainNameRaw         = `{{.Name}}`
)

const (
	scalarTypeRaw    = `{{asType .}}`
	sliceTypeRaw     = `[]{{asType .}}`
	iteratorTypeRaw  = `Iterator`
	interfaceTypeRaw = "interface{}"
	boolsTypeRaw     = `[]bool`
	boolTypeRaw      = `bool`
	intTypeRaw       = `int`
	intsTypeRaw      = `[]int`
	reflectTypeRaw   = `reflect.Type`

	// arrayTypeRaw        = `Array`
	arrayTypeRaw            = `*storage.Header`
	unaryFuncTypeRaw        = `func({{asType .}}){{asType .}} `
	unaryFuncErrTypeRaw     = `func({{asType .}}) ({{asType .}}, error)`
	reductionFuncTypeRaw    = `func(a, b {{asType .}}) {{asType .}}`
	reductionFuncTypeErrRaw = `func(a, b {{asType .}}) ({{asType .}}, error)`
	tensorTypeRaw           = `Tensor`
	splatFuncOptTypeRaw     = `...FuncOpt`
	denseTypeRaw            = `*Dense`

	testingTypeRaw = `*testing.T`
)

var (
	golinkPragma      *template.Template
	typeAnnotatedName *template.Template
	plainName         *template.Template

	scalarType       *template.Template
	sliceType        *template.Template
	iteratorType     *template.Template
	interfaceType    *template.Template
	boolsType        *template.Template
	boolType         *template.Template
	intType          *template.Template
	intsType         *template.Template
	reflectType      *template.Template
	arrayType        *template.Template
	unaryFuncType    *template.Template
	unaryFuncErrType *template.Template
	tensorType       *template.Template
	splatFuncOptType *template.Template
	denseType        *template.Template
	testingType      *template.Template
)

func init() {
	golinkPragma = template.Must(template.New("golinkPragma").Funcs(funcs).Parse(golinkPragmaRaw))
	typeAnnotatedName = template.Must(template.New("type annotated name").Funcs(funcs).Parse(typeAnnotatedNameRaw))
	plainName = template.Must(template.New("plainName").Funcs(funcs).Parse(plainNameRaw))

	scalarType = template.Must(template.New("scalarType").Funcs(funcs).Parse(scalarTypeRaw))
	sliceType = template.Must(template.New("sliceType").Funcs(funcs).Parse(sliceTypeRaw))
	iteratorType = template.Must(template.New("iteratorType").Funcs(funcs).Parse(iteratorTypeRaw))
	interfaceType = template.Must(template.New("interfaceType").Funcs(funcs).Parse(interfaceTypeRaw))
	boolsType = template.Must(template.New("boolsType").Funcs(funcs).Parse(boolsTypeRaw))
	boolType = template.Must(template.New("boolType").Funcs(funcs).Parse(boolTypeRaw))
	intType = template.Must(template.New("intTYpe").Funcs(funcs).Parse(intTypeRaw))
	intsType = template.Must(template.New("intsType").Funcs(funcs).Parse(intsTypeRaw))
	reflectType = template.Must(template.New("reflectType").Funcs(funcs).Parse(reflectTypeRaw))
	arrayType = template.Must(template.New("arrayType").Funcs(funcs).Parse(arrayTypeRaw))
	unaryFuncType = template.Must(template.New("unaryFuncType").Funcs(funcs).Parse(unaryFuncTypeRaw))
	unaryFuncErrType = template.Must(template.New("unaryFuncErrType").Funcs(funcs).Parse(unaryFuncErrTypeRaw))
	tensorType = template.Must(template.New("tensorType").Funcs(funcs).Parse(tensorTypeRaw))
	splatFuncOptType = template.Must(template.New("splatFuncOpt").Funcs(funcs).Parse(splatFuncOptTypeRaw))
	denseType = template.Must(template.New("*Dense").Funcs(funcs).Parse(denseTypeRaw))
	testingType = template.Must(template.New("*testing.T").Funcs(funcs).Parse(testingTypeRaw))
}
