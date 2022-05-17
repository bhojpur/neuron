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

type ReductionOp struct {
	OpName      string
	VecVec      string // sum(a, b []T)
	OpOfVec     string // sum([]T)
	GenericName string // sum(T, T) T
	Kinds       []reflect.Kind
	Typeclass   TypeClass
}

var reductionOps = []ReductionOp{
	{OpName: "Sum", VecVec: "VecAdd", OpOfVec: "Sum", GenericName: "Add", Typeclass: isNumber},
	{OpName: "Max", VecVec: "VecMax", OpOfVec: "SliceMax", GenericName: "Max", Typeclass: isNonComplexNumber},
	{OpName: "Min", VecVec: "VecMin", OpOfVec: "SliceMin", GenericName: "Min", Typeclass: isNonComplexNumber},
}

const reductionSpecializationRaw = `func Monotonic{{.OpName | title}}(t reflect.Type, a *storage.Header) (retVal interface{}, err error) {
	switch t {
		{{$opOfVec := .OpOfVec -}}
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		retVal = {{$opOfVec}}{{short .}}(a.{{sliceOf .}})
		return
		{{end -}}
		{{end -}}
	default:
		err = errors.Errorf("Cannot perform {{.OpName}} on %v", t)
		return
	}
}

func {{.OpName | title}}Methods(t reflect.Type)(firstFn, lasFn, defaultFn interface{}, err error) {
	{{$vecVec := .VecVec -}}
	{{$opOfVec := .OpOfVec -}}
	{{$genericName := .GenericName -}}
	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		return {{$vecVec}}{{short .}}, {{$opOfVec}}{{short .}}, {{$genericName}}{{short .}}, nil
		{{end -}}
		{{end -}}
	default:
		return nil, nil, nil, errors.Errorf("No methods found for {{.OpName}} for %v", t)
	}
}

`

var reductionSpecialization *template.Template

func init() {
	reductionSpecialization = template.Must(template.New("reduction specialization").Funcs(funcs).Parse(reductionSpecializationRaw))
}

func GenerateReductionSpecialization(f io.Writer, ak Kinds) {
	for _, op := range reductionOps {
		for _, k := range ak.Kinds {
			if !op.Typeclass(k) {
				continue
			}
			op.Kinds = append(op.Kinds, k)
		}
		reductionSpecialization.Execute(f, op)
	}
}
