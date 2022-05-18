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
	"text/template"
)

const rangeRaw = `// Range creates a ranged array with a given type. It panics if the Dtype is not supported or does not represent a naturally orderable type (strings, pointers etc)
// Do note that the range algorithm is very simple, and simply does increments or decrements of 1. This means for floating point types
// you're not able to create a range with a 0.1 increment step, and for complex number types, the imaginary part will always be 0i
func Range(dt Dtype, start, end int) interface{} {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a range that is negative in size")
	}
	switch dt.Kind(){
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
		{{if isRangeable . -}}
	case reflect.{{reflectKind .}}:
		{{if hasPrefix .String "float" -}}
			return vec{{short . | lower}}.Range(start, end)
		{{else -}}
			retVal := make([]{{asType .}}, size)
			{{if eq .String "complex64" -}}
			for i, v := 0, complex(float32(start), float32(0.0)); i < size; i++ {
			{{else if eq .String "complex128" -}}
			for i, v := 0, complex(float64(start), float64(0.0)); i < size; i++ {
			{{else -}}
			for i, v := 0, {{asType .}}(start); i < size; i++ {
			{{end -}}
				retVal[i] = v
				if incr {
					v++
				} else{
					v--
				}
			}
			return retVal
		{{end -}}
		{{end -}}
		{{end -}}
	{{end -}}
	default:
		err := errors.Errorf("Unrangeable Type %v", dt)
		panic(err)
	}
}
`

const randomRaw = `// Random creates an array of random numbers of the given type.
// For complex Dtypes, the imaginary component will be 0.
//
// This function is only useful in cases where the randomness is not vital. 
func Random(dt Dtype, size int) interface{} {
	r := rand.New(rand.NewSource(1337))
	switch dt.Kind() {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		retVal := make([]{{asType .}}, size)
		for i := range retVal {
			retVal[i] = {{if hasPrefix .String "int" -}}
				{{asType .}}(r.Int())
			{{else if hasPrefix .String "uint" -}}
				{{asType .}}(r.Uint32())
			{{else if hasPrefix .String "complex64" -}}
				complex(r.Float32(), float32(0))
			{{else if hasPrefix .String "complex128" -}}
				complex(r.Float64(), float64(0))
			{{else if eq .String "float64" -}}
				rand.NormFloat64()
			{{else if eq .String "float32" -}}
				float32(r.NormFloat64())
			{{end -}}
		}
		return retVal
	{{end -}}
	{{end -}}
	}
	panic("unreachable")
}
`

var (
	Range  *template.Template
	Random *template.Template
)

func init() {
	Range = template.Must(template.New("Range").Funcs(funcs).Parse(rangeRaw))
	Random = template.Must(template.New("Random").Funcs(funcs).Parse(randomRaw))
}

func GenerateUtils(f io.Writer, generic Kinds) {
	importStmt := `
import (
	"math/rand"
	"reflect"

	"github.com/pkg/errors"
	"github.com/bhojpur/neuron/pkg/vector/vecf32"
	"github.com/bhojpur/neuron/pkg/vector/vecf64"
)
`
	f.Write([]byte(importStmt))
	Range.Execute(f, generic)
	Random.Execute(f, generic)
}
