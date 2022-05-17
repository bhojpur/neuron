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

const onesTestsRaw = `var onesTests = []struct {
	of Dtype
	shape Shape
	correct interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ {{asType . | title | strip}},  ScalarShape(), {{asType .}}(1)},
	{ {{asType . | title | strip}},  Shape{2,2}, []{{asType .}}{1,1,1,1}},
	{{end -}}
	{{end -}}
	{Bool, ScalarShape(), true},
	{Bool, Shape{2,2}, []bool{true, true, true, true}},
}

func TestOnes(t *testing.T){
	assert := assert.New(t)
	for _, ot := range onesTests{
		T := Ones(ot.of, ot.shape...)
		assert.True(ot.shape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())
	}
}
`

const eyeTestsRaw = `// yes, it's a pun on eye tests, stop asking and go see your optometrist
var eyeTests = []struct{
	E Dtype
	R, C, K int


	correct interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ {{asType . | title | strip}}, 4,4, 0, []{{asType .}}{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
	{ {{asType . | title | strip}}, 4,4, 1, []{{asType .}}{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 2, []{{asType .}}{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 3, []{{asType .}}{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 4, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -1, []{{asType .}}{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
	{ {{asType . | title | strip}}, 4,4, -2, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -3, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -4, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,5, 0, []{{asType .}}{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}},
	{ {{asType . | title | strip}}, 4,5, 1, []{{asType .}}{0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1}},
	{ {{asType . | title | strip}}, 4,5, -1, []{{asType .}}{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0}},
	{{end -}}
	{{end -}}
}

func TestI(t *testing.T){
	assert := assert.New(t)
	var T Tensor

	for i, it := range eyeTests {
		T = I(it.E, it.R, it.C, it.K)
		assert.True(Shape{it.R, it.C}.Eq(T.Shape()))
		assert.Equal(it.correct, T.Data(), "Test %d-R: %d, C: %d K: %d", i, it.R, it.C, it.K)
	}

}
`

var (
	onesTests *template.Template
	eyeTests  *template.Template
)

func init() {
	onesTests = template.Must(template.New("onesTest").Funcs(funcs).Parse(onesTestsRaw))
	eyeTests = template.Must(template.New("eyeTest").Funcs(funcs).Parse(eyeTestsRaw))
}

func GenerateDenseConsTests(f io.Writer, generic Kinds) {
	onesTests.Execute(f, generic)
	eyeTests.Execute(f, generic)
}
