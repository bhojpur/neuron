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

const asSliceRaw = `func (h *Header) {{asType . | strip | title}}s() []{{asType .}} {return (*(*[]{{asType .}})(unsafe.Pointer(&h.Raw)))[:h.TypedLen({{short . | unexport}}Type):h.TypedLen({{short . | unexport}}Type)]}
`

const setBasicRaw = `func (h *Header) Set{{short . }}(i int, x {{asType . }}) { h.{{sliceOf .}}[i] = x }
`

const getBasicRaw = `func (h *Header) Get{{short .}}(i int) {{asType .}} { return h.{{lower .String | clean | strip | title }}s()[i]}
`

const getRaw = `// Get returns the ith element of the underlying array of the *Dense tensor.
func (a *array) Get(i int) interface{} {
	switch a.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		return a.{{getOne .}}(i)
		{{end -}};
	{{end -}}
	default:
		val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

`
const setRaw = `// Set sets the value of the underlying array at the index i.
func (a *array) Set(i int, x interface{}) {
	switch a.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		xv := x.({{asType .}})
		a.{{setOne .}}(i, xv)
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

`

const memsetRaw = `// Memset sets all values in the array.
func (a *array) Memset(x interface{}) error {
	switch a.t {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		if xv, ok := x.({{asType .}}); ok {
			data := a.{{sliceOf .}}
			for i := range data{
				data[i] = xv
			}
			return nil
		}

		{{end -}}
	{{end -}}
	}

	xv := reflect.ValueOf(x)
	l := a.Len()
	for i := 0; i < l; i++ {
		val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
	return nil
}
`

const arrayEqRaw = ` // Eq checks that any two arrays are equal
func (a array) Eq(other interface{}) bool {
	if oa, ok := other.(*array); ok {
		if oa.t != a.t {
			return false
		}

		if oa.Len() != a.Len() {
			return false
		}
		/*
		if oa.C != a.C {
			return false
		}
		*/

		// same exact thing
		if uintptr(unsafe.Pointer(&oa.Header.Raw[0])) == uintptr(unsafe.Pointer(&a.Header.Raw[0])){
			return true
		}

		switch a.t.Kind() {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case reflect.{{reflectKind .}}:
			for i, v := range a.{{sliceOf .}} {
				if oa.{{getOne .}}(i) != v {
					return false
				}
			}
			{{end -}}
		{{end -}}
		default:
			for i := 0; i < a.Len(); i++{
				if !reflect.DeepEqual(a.Get(i), oa.Get(i)){
					return false
				}
			}
		}
		return true
	}
	return false
}`

const copyArrayIterRaw = `func copyArrayIter(dst, src array, diter, siter Iterator) (count int, err error){
	if dst.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil {
		return copyArray(dst, src), nil
	}

	if (diter != nil && siter == nil) || (diter == nil && siter != nil) {
		return 0, errors.Errorf("Cannot copy array when only one iterator was passed in")
	}

	k := dest.t.Kind()
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = diter.NextValidity(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		if j, validj, err = siter.NextValidity(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		switch k {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case reflect.{{reflectKind .}}:
			dest.{{setOne .}}(i, src.{{getOne .}}(j))
			{{end -}}
		{{end -}}
		default:
			dest.Set(i, src.Get(j))
		}
		count++
	}

}
`

const memsetIterRaw = `
func (a *array) memsetIter(x interface{}, it Iterator) (err error) {
	var i int
	switch a.t{
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		xv, ok := x.({{asType .}})
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.{{sliceOf .}}
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			data[i] = xv
		}
		err = handleNoOp(err)
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
		err = handleNoOp(err)
	}
	return
}

`

const zeroIterRaw = `func (a *array) zeroIter(it Iterator) (err error){
	var i int
	switch a.t {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		data := a.{{sliceOf .}}
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			data[i] = {{if eq .String "bool" -}}
				false
			{{else if eq .String "string" -}}""
			{{else if eq .String "unsafe.Pointer" -}}nil
			{{else -}}0{{end -}}
		}
		err = handleNoOp(err)
		{{end -}}
	{{end -}}
	default:
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(a.t))
		}
		err = handleNoOp(err)
	}
	return
}
`

const reflectConstTemplateRaw = `var (
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
			{{short . | unexport}}Type = reflect.TypeOf({{asType .}}({{if eq .String "bool" -}} false {{else if eq .String "string" -}}"" {{else if eq .String "unsafe.Pointer" -}}nil {{else -}}0{{end -}}))
		{{end -}}
	{{end -}}
)`

var (
	AsSlice     *template.Template
	SimpleSet   *template.Template
	SimpleGet   *template.Template
	Get         *template.Template
	Set         *template.Template
	Memset      *template.Template
	MemsetIter  *template.Template
	Eq          *template.Template
	ZeroIter    *template.Template
	ReflectType *template.Template
)

func init() {
	AsSlice = template.Must(template.New("AsSlice").Funcs(funcs).Parse(asSliceRaw))
	SimpleSet = template.Must(template.New("SimpleSet").Funcs(funcs).Parse(setBasicRaw))
	SimpleGet = template.Must(template.New("SimpleGet").Funcs(funcs).Parse(getBasicRaw))
	Get = template.Must(template.New("Get").Funcs(funcs).Parse(getRaw))
	Set = template.Must(template.New("Set").Funcs(funcs).Parse(setRaw))
	Memset = template.Must(template.New("Memset").Funcs(funcs).Parse(memsetRaw))
	MemsetIter = template.Must(template.New("MemsetIter").Funcs(funcs).Parse(memsetIterRaw))
	Eq = template.Must(template.New("ArrayEq").Funcs(funcs).Parse(arrayEqRaw))
	ZeroIter = template.Must(template.New("Zero").Funcs(funcs).Parse(zeroIterRaw))
	ReflectType = template.Must(template.New("ReflectType").Funcs(funcs).Parse(reflectConstTemplateRaw))
}

func GenerateArrayMethods(f io.Writer, ak Kinds) {
	Set.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Get.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Memset.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	MemsetIter.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Eq.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	ZeroIter.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
}

func GenerateHeaderGetSet(f io.Writer, ak Kinds) {
	for _, k := range ak.Kinds {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}

func GenerateReflectTypes(f io.Writer, ak Kinds) {
	ReflectType.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
}
