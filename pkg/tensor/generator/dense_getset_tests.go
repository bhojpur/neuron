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
	"reflect"
	"text/template"
)

type testData struct {
	Kind      reflect.Kind
	TestData0 []interface{}
	Set       interface{}
	Correct   []interface{}
}

func makeTests(generic Kinds) []testData {
	retVal := make([]testData, 0)
	for _, k := range generic.Kinds {
		if isParameterized(k) {
			continue
		}

		td := testData{Kind: k}

		data := make([]interface{}, 6)
		correct := make([]interface{}, 6)

		switch {
		case isRangeable(k):
			raw := []int{0, 1, 2, 3, 4, 5}
			for i := range data {
				data[i] = raw[i]
				correct[i] = 45
			}
			td.Set = 45
		case k == reflect.Bool:
			raw := []bool{true, false, true, false, true, false}
			for i := range data {
				data[i] = raw[i]
				correct[i] = false
			}
			td.Set = false
		case k == reflect.String:
			raw := []string{"\"zero\"", "\"one\"", "\"two\"", "\"three\"", "\"four\"", "\"five\""}
			for i := range data {
				data[i] = raw[i]
				correct[i] = "\"HELLO WORLD\""
			}
			td.Set = "\"HELLO WORLD\""
		default:
			continue
		}
		td.TestData0 = data
		td.Correct = correct
		retVal = append(retVal, td)

	}
	return retVal
}

func makeZeroTests(generic Kinds) []testData {
	retVal := make([]testData, 0)
	for _, k := range generic.Kinds {
		if isParameterized(k) {
			continue
		}

		td := testData{Kind: k}

		data := make([]interface{}, 6)
		correct := make([]interface{}, 6)

		switch {
		case isRangeable(k):
			raw := []int{0, 1, 2, 3, 4, 5}
			for i := range data {
				data[i] = raw[i]
				correct[i] = 0
			}
		case k == reflect.Bool:
			raw := []bool{true, false, true, false, true, false}
			for i := range data {
				data[i] = raw[i]
				correct[i] = false
			}
		case k == reflect.String:
			raw := []string{"\"zero\"", "\"one\"", "\"two\"", "\"three\"", "\"four\"", "\"five\""}
			for i := range data {
				data[i] = raw[i]
				correct[i] = "\"\""
			}
		default:
			continue
		}
		td.TestData0 = data
		td.Correct = correct
		retVal = append(retVal, td)

	}
	return retVal
}

const getTestRaw = `var denseSetGetTests = []struct {
	of Dtype
	data interface{} 
	set interface{}

	correct []interface{}
}{
	{{range . -}}
	{{$k := .Kind -}}
	{ {{title .Kind.String | strip}}, []{{.Kind.String | clean}}{ {{range .TestData0 -}}{{printf "%v" .}}, {{end -}} }, {{printf "%v" .Set}}, []interface{}{ {{range .TestData0 -}} {{$k}}({{printf "%v" .}}), {{end -}} }},
	{{end -}}
}

func TestDense_setget(t *testing.T) {
	assert := assert.New(t)
	for _, gts := range denseSetGetTests {
		T := New(Of(gts.of), WithShape(len(gts.correct)))
		for i, v := range gts.correct {
			T.Set(i, v)
			got := T.Get(i)
			assert.Equal(v, got)
		}
	}
}

`

const memsetTestRaw = `var denseMemsetTests = []struct{
	of Dtype
	data interface{}
	val interface{}
	shape Shape

	correct interface{}
}{
	{{range . -}}
	{{$val := .Set -}}
	{{$k := .Kind -}}
	{ {{title .Kind.String | strip}}, []{{asType .Kind}}{ {{range .TestData0 -}}{{printf "%v" .}}, {{end -}} }, {{asType .Kind}}({{$val}}), Shape{2,3}, []{{asType .Kind}}{ {{range .Correct}} {{printf "%v" .}}, {{end -}} } }, 
	{{end -}}
}

func TestDense_memset(t *testing.T){
	assert := assert.New(t)
	for _, mts := range denseMemsetTests {
		T := New(Of(mts.of), WithShape(mts.shape...))
		T.Memset(mts.val)
		assert.Equal(mts.correct, T.Data())

		T = New(Of(mts.of), WithShape(mts.shape...), WithBacking(mts.data))
		T2, _ := T.Slice(nil)
		T2.Memset(mts.val)
		assert.Equal(mts.correct, T2.Data())
	}
}
`

const zeroTestRaw = `var denseZeroTests = []struct{
	of Dtype
	data interface{}

	correct interface{}
}{
	{{range . -}}
	{{$val := .Set -}}
	{{$k := .Kind -}}
	{ {{title .Kind.String | strip}}, []{{asType .Kind}}{ {{range .TestData0 -}}{{printf "%v" .}}, {{end -}} }, []{{asType .Kind}}{ {{range .Correct}} {{printf "%v" .}}, {{end -}} } }, 
	{{end -}}
}

func TestDense_Zero(t *testing.T) {
	assert := assert.New(t)
	for _, mts := range denseZeroTests {
		
		typ := reflect.TypeOf(mts.data)
		val := reflect.ValueOf(mts.data)
		data := reflect.MakeSlice(typ, val.Len(), val.Cap())
		reflect.Copy(data, val)	

		T := New(Of(mts.of), WithBacking(data.Interface()))
		T.Zero()
		assert.Equal(mts.correct, T.Data())

		T = New(Of(mts.of),  WithBacking(mts.data))
		T2, _ := T.Slice(nil)
		T2.Zero()
		assert.Equal(mts.correct, T2.Data())
	}	
}
`

const denseEqTestRaw = `func TestDense_Eq(t *testing.T) {
	eqFn := func(q *Dense) bool{
		a := q.Clone().(*Dense)
		if !q.Eq(a) {
			t.Error("Expected a clone to be exactly equal")
			return false
		}
		a.Zero()

		// Bools are excluded because the probability of having an array of all false is very high
		if q.Eq(a)  && a.len() > 3 && a.Dtype() != Bool {
			t.Errorf("a %v", a.Data())
			t.Errorf("q %v", q.Data())
			t.Error("Expected *Dense to be not equal")
			return false
		}
		return true
	}
	if err := quick.Check(eqFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Failed to perform equality checks")
	}
}`

var (
	GetTest    *template.Template
	MemsetTest *template.Template
	ZeroTest   *template.Template
)

func init() {
	GetTest = template.Must(template.New("GetTest").Funcs(funcs).Parse(getTestRaw))
	MemsetTest = template.Must(template.New("MemsetTest").Funcs(funcs).Parse(memsetTestRaw))
	ZeroTest = template.Must(template.New("ZeroTest").Funcs(funcs).Parse(zeroTestRaw))
}

func GenerateDenseGetSetTests(f io.Writer, generic Kinds) {
	tests := makeTests(generic)
	GetTest.Execute(f, tests)
	fmt.Fprintf(f, "\n\n")
	MemsetTest.Execute(f, tests)
	fmt.Fprintf(f, "\n\n")
	ZeroTest.Execute(f, makeZeroTests(generic))
	fmt.Fprintf(f, "\n%v\n", denseEqTestRaw)
}
