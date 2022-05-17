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

const anyToF64sRaw = `func anyToFloat64s(x interface{}) (retVal []float64) {
	switch xt := x.(type) {
	{{range  .Kinds -}}
	{{if isNumber . -}}
	case []{{asType .}}:
		{{if eq .String "float64" -}}
		{{else if eq .String "float32" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case math32.IsNaN(v):
					retVal[i] = math.NaN()
				case math32.IsInf(v, 1):
					retVal[i] = math.Inf(1)
				case math32.IsInf(v, -1):
					retVal[i] = math.Inf(-1)
				default:
					retVal[i] = float64(v)
				}
			}
		{{else if eq .String "complex64" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case cmplx.IsNaN(complex128(v)):
					retVal[i] = math.NaN()
				case cmplx.IsInf(complex128(v)):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = float64(real(v))
				}
			}
		{{else if eq .String "complex128" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case cmplx.IsNaN(v):
					retVal[i] = math.NaN()
				case cmplx.IsInf(v):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = real(v)
				}
			}
		{{else -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				retVal[i]=  float64(v)
			}
		{{end -}}
		return {{if eq .String "float64"}}xt{{end}}
	{{end -}}
	{{end -}}
	}
	panic("Unreachable")
}
`

const qcGenraw = `func randomQC(a Tensor, r *rand.Rand) {
	switch a.Dtype() {
	{{range .Kinds -}}
	{{if isParameterized . -}}
	{{else -}}
	case {{reflectKind . -}}:
		s := a.Data().([]{{asType .}})
		for i := range s {
			{{if hasPrefix .String "uint" -}}
				s[i] = {{asType .}}(r.Uint32())
			{{else if hasPrefix .String "int" -}}
				s[i] = {{asType .}}(r.Int())
			{{else if eq .String "float64" -}}
				s[i] = r.Float64()
			{{else if eq .String "float32" -}}
				s[i] = r.Float32()
			{{else if eq .String "complex64" -}}
				s[i] = complex(r.Float32(), r.Float32())
			{{else if eq .String "complex128" -}}
				s[i] = complex(r.Float64(), r.Float64())
			{{else if eq .String "bool" -}}
				s[i] = randomBool()
			{{else if eq .String "string" -}}
				s[i] = randomString()
			{{else if eq .String "unsafe.Pointer" -}}
				s[i] = nil
			{{end -}}	
		}
	{{end -}}
	{{end -}}
	}
}
`

const testQCRaw = `type QCDense{{short .}} struct {
	*Dense 
}
func (*QCDense{{short .}}) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]{{asType .}}, size)
	for i := range s {
		{{if hasPrefix .String "uint" -}}
			s[i] = {{asType .}}(r.Uint32())
		{{else if hasPrefix .String "int" -}}
			s[i] = {{asType .}}(r.Int())
		{{else if eq .String "float64" -}}
			s[i] = r.Float64()
		{{else if eq .String "float32" -}}
			s[i] = r.Float32()
		{{else if eq .String "complex64" -}}
			s[i] = complex(r.Float32(), r.Float32())
		{{else if eq .String "complex128" -}}
			s[i] = complex(r.Float64(), r.Float64())
		{{else if eq .String "bool" -}}
			s[i] = randomBool()
		{{else if eq .String "string" -}}
			s[i] = randomString()
		{{else if eq .String "unsafe.Pointer" -}}
			s[i] = nil
		{{end -}}
	}
	d := recycledDense({{asType . | title | strip}}, Shape{size}, WithBacking(s))
	q := new(QCDense{{short .}})
	q.Dense = d
	return reflect.ValueOf(q)
}
`

const identityFnsRaw = `func identity{{short .}}(a {{asType .}}) {{asType .}}{return a}
`
const mutateFnsRaw = `func mutate{{short .}}(a {{asType . }}){{asType .}} { {{if isNumber . -}}return 1}
{{else if eq .String "bool" -}}return true }
{{else if eq .String "string" -}}return "Hello World"}
{{else if eq .String "uintptr" -}}return 0xdeadbeef}
{{else if eq .String "unsafe.Pointer" -}}return unsafe.Pointer(uintptr(0xdeadbeef))} 
{{end -}} 
`

const identityValsRaw = `func identityVal(x int, dt Dtype) interface{} {
	switch dt {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		return {{asType .}}(x)
		{{end -}}
	case Complex64:
		var c complex64
		if x == 0 {
			return c
		}
		c = 1
		return c
	case Complex128:
		var c complex128
		if x == 0 {
			return c
		}
		c = 1
		return c
	case Bool:
		if x == 0 {
			return false
		}
		return true
	case String:
		if x == 0 {
			return ""
		}
		return fmt.Sprintf("%v", x)
	default:
		return x
	}
}`

const threewayEqualityRaw = `func threewayEq(a, b, c interface{}) bool {
	switch at := a.(type){
		{{range .Kinds -}}
	case []{{asType .}}:
		bt := b.([]{{asType .}})
		ct := c.([]{{asType .}})

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
		{{end -}}
		{{range .Kinds -}}
	case {{asType .}}:
		bt := b.({{asType .}})
		ct := c.({{asType .}})
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
		{{end -}}
	}

	return false
}
`

var (
	anyToF64s        *template.Template
	qcGen            *template.Template
	testQC           *template.Template
	identityFns      *template.Template
	mutateFns        *template.Template
	identityVals     *template.Template
	threewayEquality *template.Template
)

func init() {
	qcGen = template.Must(template.New("QCGen").Funcs(funcs).Parse(qcGenraw))
	testQC = template.Must(template.New("testQCs").Funcs(funcs).Parse(testQCRaw))
	anyToF64s = template.Must(template.New("anyToF64s").Funcs(funcs).Parse(anyToF64sRaw))
	identityFns = template.Must(template.New("identityFn").Funcs(funcs).Parse(identityFnsRaw))
	mutateFns = template.Must(template.New("mutateFns").Funcs(funcs).Parse(mutateFnsRaw))
	identityVals = template.Must(template.New("identityVal").Funcs(funcs).Parse(identityValsRaw))
	threewayEquality = template.Must(template.New("threeway").Funcs(funcs).Parse(threewayEqualityRaw))
}

func GenerateTestUtils(f io.Writer, ak Kinds) {
	anyToF64s.Execute(f, ak)
	fmt.Fprintf(f, "\n")
	ak2 := Kinds{Kinds: filter(ak.Kinds, isNonComplexNumber)}
	identityVals.Execute(f, ak2)
	fmt.Fprintf(f, "\n")
	ak3 := Kinds{Kinds: filter(ak.Kinds, isNumber)}
	threewayEquality.Execute(f, ak3)
	fmt.Fprintf(f, "\n")
	for _, k := range ak.Kinds {
		if !isParameterized(k) {
			identityFns.Execute(f, k)
		}
	}
	for _, k := range ak.Kinds {
		if !isParameterized(k) {
			mutateFns.Execute(f, k)
		}
	}
	fmt.Fprintf(f, "\n")
	// for _, k := range ak.Kinds {
	// 	if !isParameterized(k) {
	// 		testQC.Execute(f, k)
	// 		fmt.Fprint(f, "\n")
	// 	}
	// }

}
