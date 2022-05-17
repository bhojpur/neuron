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

const reflectBasedReduceRaw = `func ReduceRef(f reflect.Value, fnT reflect.Type, def reflect.Value, l *Dense) interface{} {
	retVal := def
	if l.len() == 0 {
		return retVal.Interface()
	}

	args := make([]reflect.Value, 0, fnT.NumIn())
	for i := 0; i < l.len(); i++ {
		v := reflect.ValueOf(l.Get(i))
		args = append(args, retVal)
		args = append(args, v)
		retVal = f.Call(args)[0]
		args = args[:0]
	}
	return retVal.Interface()
}

`
const genericReduceRaw = `func Reduce{{short .}}(f func(a, b {{asType .}}) {{asType .}}, def {{asType .}}, l ...{{asType .}}) (retVal {{asType .}}){
	retVal = def
	if len(l) == 0 {
		return 
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

`

const genericSumRaw = `func Sum{{short .}}(a []{{asType .}}) {{asType .}}{ 
	var retVal {{asType .}}
	a = a[:len(a)]
	for _, v := range a {
		retVal += v
	}
	return retVal
}
`

const genericProdRaw = `func Prod{{short .}}(a []{{asType .}}) {{asType .}} { 
	if len(a) == 0 {
		return 0
	}
	var retVal {{asType .}} = 1
	a = a[:len(a)]
	for _, v := range a {
		retVal *= v
	}
	return retVal
}
`

const genericSliceMinMaxRaw = `func SliceMin{{short .}}(a []{{asType .}}) {{asType .}}{
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return Reduce{{short .}}(Min{{short .}}, a[0], a[1:]...)
}

func SliceMax{{short .}}(a []{{asType .}}) {{asType .}}{
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return Reduce{{short .}}(Max{{short .}}, a[0], a[1:]...)
}

`

const genericReduce0Raw = `func reduceFirst{{short .}}(data, retVal []{{asType .}}, split, size int, fn func(a, b []{{asType .}})) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size - 1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirst{{short .}}(data, retVal []{{asType .}}, split, size int, fn func(a, b {{asType .}}){{asType .}} ){
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size - 1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
`

const genericReduce0ParRaw = `func reduceFirst{{short .}}(data, retVal []{{asType .}}, split, size int, fn func(a, b {{asType .}}){{asType .}}) {
	start := split
	var wg sync.Waitgroup
	for i := 0; i < size - 1; i++ {
		wg.Add(1)
		go func(sp, st int) {
			for j := 0; j < sp; j++ {
				retVal[j] = fn(retVal[j], data[j+start])
			}
		}(split, start, &wg)
		start += split
	}
}

`

const genericReduceLastRaw = `func reduceLast{{short .}}(a, retVal []{{asType .}}, dimSize int, defaultValue {{asType .}}, fn func(a []{{asType .}}){{asType .}}) {
	var at int
	for start := 0; start <= len(a) - dimSize; start += dimSize {
		r := fn(a[start:start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLast{{short .}}(a, retVal []{{asType .}}, dimSize int, defaultValue {{asType .}}, fn func({{asType .}}, {{asType .}}){{asType .}}) {
	var at int 
	for start := 0; start <= len(a) - dimSize; start += dimSize {
		r := Reduce{{short .}}(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

`

const genericReduceDefaultRaw = `func reduceDefault{{short .}}(data, retVal []{{asType .}}, dim0, dimSize, outerStride, stride, expected int, fn func(a,b {{asType .}}){{asType .}}) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int 
		for j := 0; j < expected; j++ {
			writeTo := i * expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k * stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

`

var (
	genericReduce      *template.Template
	genericSum         *template.Template
	genericProd        *template.Template
	genericSliceMinMax *template.Template

	genericReduce0       *template.Template
	genericReduceLast    *template.Template
	genericReduceDefault *template.Template
)

func init() {
	genericReduce = template.Must(template.New("genericReduce").Funcs(funcs).Parse(genericReduceRaw))
	genericSum = template.Must(template.New("genericSum").Funcs(funcs).Parse(genericSumRaw))
	genericProd = template.Must(template.New("genericProd").Funcs(funcs).Parse(genericProdRaw))
	genericSliceMinMax = template.Must(template.New("genericSliceMinMax").Funcs(funcs).Parse(genericSliceMinMaxRaw))
	genericReduce0 = template.Must(template.New("genericReduce0").Funcs(funcs).Parse(genericReduce0Raw))
	genericReduceLast = template.Must(template.New("genericReduceLast").Funcs(funcs).Parse(genericReduceLastRaw))
	genericReduceDefault = template.Must(template.New("genericReduceDefault").Funcs(funcs).Parse(genericReduceDefaultRaw))
}

func GenerateGenericReduce(f io.Writer, generic Kinds) {
	// fmt.Fprintln(f, reflectBasedReduceRaw)
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			genericReduce.Execute(f, k)
		}
	}

	for _, k := range filter(generic.Kinds, isNumber) {
		genericSum.Execute(f, k)

	}
	for _, k := range filter(generic.Kinds, isNumber) {
		genericProd.Execute(f, k)
	}
	fmt.Fprintf(f, "\n")

	for _, k := range filter(generic.Kinds, isOrd) {
		if isNumber(k) {
			genericSliceMinMax.Execute(f, k)
		}
	}

	for _, k := range filter(generic.Kinds, isNotParameterized) {
		genericReduce0.Execute(f, k)
	}

	for _, k := range filter(generic.Kinds, isNotParameterized) {
		genericReduceLast.Execute(f, k)
	}

	for _, k := range filter(generic.Kinds, isNotParameterized) {
		genericReduceDefault.Execute(f, k)
	}
}
