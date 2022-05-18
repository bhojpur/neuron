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

const testDenseReduceRaw = `var denseReductionTests = []struct {
	of Dtype
	fn interface{}
	def interface{}
	axis int

	correct interface{}
	correctShape Shape
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	// {{.}}
	{ {{asType . | title}}, execution.Add{{short .}}, {{asType .}}(0), 0, []{{asType .}}{6, 8, 10, 12, 14, 16}, Shape{3,2} },
	{ {{asType . | title}}, execution.Add{{short .}}, {{asType .}}(0), 1, []{{asType .}}{6, 9, 24, 27}, Shape{2, 2}},
	{ {{asType . | title}}, execution.Add{{short .}}, {{asType .}}(0), 2, []{{asType .}}{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	{{end -}}
	{{end -}}
}

func TestDense_Reduce(t *testing.T){
	assert := assert.New(t)
	for _, drt := range denseReductionTests {
		T := New(WithShape(2,3,2), WithBacking(Range(drt.of, 0, 2*3*2)))
		T2, err := T.Reduce(drt.fn, drt.axis, drt.def, )
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(drt.correctShape.Eq(T2.Shape()))
		assert.Equal(drt.correct, T2.Data())

		// stupids:
		_, err = T.Reduce(drt.fn, 1000, drt.def,)
		assert.NotNil(err)

		// wrong function type
		var f interface{}
		f = func(a, b float64)float64{return 0}
		if drt.of == Float64 {
			f = func(a, b int)int{return 0}
		}

		_, err = T.Reduce(f, 0, drt.correct)
		assert.NotNil(err)

		// wrong default value type
		var def2 interface{}
		def2 = 3.14
		if drt.of == Float64 {
			def2 = int(1)
		}

		_, err = T.Reduce(drt.fn, 3, def2) // only last axis requires a default value
		assert.NotNil(err)
	}
}
`

var (
	testDenseReduce *template.Template
)

func init() {
	testDenseReduce = template.Must(template.New("testDenseReduce").Funcs(funcs).Parse(testDenseReduceRaw))
}

func GenerateDenseReductionTests(f io.Writer, generic Kinds) {
	importStmt := `
import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/bhojpur/neuron/pkg/tensor/internal/execution"
)
`
	f.Write([]byte(importStmt))
	testDenseReduce.Execute(f, generic)
}
