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

const checkNativeiterable = `func checkNativeIterable(t *Dense, dims int, dt Dtype) error {
	// checks:
	if !t.IsNativelyAccessible() {
		return errors.Errorf("Cannot convert *Dense to *mat.Dense. Data is inaccessible")
	}

	if t.Shape().Dims() != dims {
		return errors.Errorf("Cannot convert *Dense to native iterator. Expected number of dimension: %d, T has got %d dimensions (Shape: %v)", dims, t.Dims(), t.Shape())
	}

	if t.F() || t.RequiresIterator() {
		return errors.Errorf("Not yet implemented: native matrix for colmajor or unpacked matrices")
	}

	if t.Dtype() != dt {
		return errors.Errorf("Conversion to native iterable only works on %v. Got %v", dt, t.Dtype())
	}

	return nil
}
`

const nativeIterRaw = `// Vector{{short .}} converts a *Dense into a []{{asType .}}
// If the *Dense does not represent a vector of the wanted type, it will return
// an error.
func Vector{{short .}}(t *Dense) (retVal []{{asType .}}, err error) {
	if err = checkNativeIterable(t, 1, {{reflectKind .}}); err != nil {
		return nil, err
	}
	return t.{{sliceOf .}}, nil
}

// Matrix{{short .}} converts a  *Dense into a [][]{{asType .}}
// If the *Dense does not represent a matrix of the wanted type, it
// will return an error.
func Matrix{{short .}}(t *Dense) (retVal [][]{{asType .}}, err error) {
	if err = checkNativeIterable(t, 2, {{reflectKind .}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .}}
	shape := t.Shape()
	strides := t.Strides()

	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]
	retVal = make([][]{{asType .}}, rows)
	for i := range retVal {
		start := i * rowStride
		retVal[i] = make([]{{asType .}}, 0)
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i]))
		hdr.Data = uintptr(unsafe.Pointer(&data[start]))
		hdr.Cap = cols
		hdr.Len = cols
	}
	return
}

// Tensor3{{short .}} converts a *Dense into a  [][][]{{asType .}}.
// If the *Dense does not represent a 3-tensor of the wanted type, it will return an error.
func Tensor3{{short .}}(t *Dense) (retVal [][][]{{asType .}}, err error) {
	if err = checkNativeIterable(t, 3, {{reflectKind .}}); err != nil {
		return nil, err
	}

	data := t.{{sliceOf .}}
	shape := t.Shape()
	strides := t.Strides()

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]
	retVal = make([][][]{{asType .}}, layers)
	for i := range retVal {
		retVal[i] = make([][]{{asType .}}, rows)
		for j := range retVal[i] {
			retVal[i][j] = make([]{{asType .}}, 0)
			start := i*layerStride + j*rowStride
			hdr := (*reflect.SliceHeader)(unsafe.Pointer(&retVal[i][j]))
			hdr.Data = uintptr(unsafe.Pointer(&data[start]))
			hdr.Cap = cols
			hdr.Len = cols
		}
	}
	return
}
`

const nativeIterTestRaw = `func Test_Vector{{short .}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable . -}}
	T = New(WithBacking(Range({{reflectKind .}}, 0, 6)), WithShape(6))
	{{else -}}
	T = New(Of({{reflectKind .}}), WithShape(6))
	{{end -}}
	it, err := Vector{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_Matrix{{short .}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable . -}}
	T = New(WithBacking(Range({{reflectKind .}}, 0, 6)), WithShape(2, 3))
	{{else -}}
	T = New(Of({{reflectKind .}}), WithShape(2, 3))
	{{end -}}
	it, err := Matrix{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3{{short .}}(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	{{if isRangeable . -}}
	T = New(WithBacking(Range({{reflectKind .}}, 0, 24)), WithShape(2, 3, 4))
	{{else -}}
	T = New(Of({{reflectKind .}}), WithShape(2, 3, 4))
	{{end -}}
	it, err := Tensor3{{short .}}(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}
`

var (
	NativeIter     *template.Template
	NativeIterTest *template.Template
)

func init() {
	NativeIter = template.Must(template.New("NativeIter").Funcs(funcs).Parse(nativeIterRaw))
	NativeIterTest = template.Must(template.New("NativeIterTest").Funcs(funcs).Parse(nativeIterTestRaw))
}

func GenerateNativeIterators(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnqualifiedTensor)
	fmt.Fprintf(f, "%v\n", checkNativeiterable)
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		fmt.Fprintf(f, "/* Native Iterables for %v */\n\n", k)
		NativeIter.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}

func GenerateNativeIteratorTests(f io.Writer, ak Kinds) {
	fmt.Fprintf(f, importUnqualifiedTensor)
	ks := filter(ak.Kinds, isSpecialized)
	for _, k := range ks {
		NativeIterTest.Execute(f, k)
		fmt.Fprint(f, "\n\n")
	}
}
