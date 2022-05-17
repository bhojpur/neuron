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

const copySlicedRaw = `func copySliced(dest *Dense, dstart, dend int, src *Dense, sstart, send int) int{
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if src.IsMasked(){
		mask:=dest.mask
		if cap(dest.mask) < dend{
			mask = make([]bool, dend)
		}
		copy(mask, dest.mask)
		dest.mask=mask
		copy(dest.mask[dstart:dend], src.mask[sstart:send])
	}
	switch dest.t {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case {{reflectKind .}}:
		return copy(dest.{{sliceOf .}}[dstart:dend], src.{{sliceOf .}}[sstart:send])
		{{end -}}
	{{end -}}
	default:
		dv := reflect.ValueOf(dest.v)
		dv = dv.Slice(dstart, dend)
		sv := reflect.ValueOf(src.v)
		sv = sv.Slice(sstart, send)
		return reflect.Copy(dv, sv)
	}	
}
`

const copyIterRaw = `func copyDenseIter(dest, src *Dense, diter, siter *FlatIterator) (int, error) {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil && !dest.IsMaterializable() && !src.IsMaterializable() {
		return copyDense(dest, src), nil
	}

	if diter == nil {
		diter = newFlatIterator(&dest.AP)	
	}
	if siter == nil {
		siter = newFlatIterator(&src.AP)
	}
	
	isMasked:= src.IsMasked()
	if isMasked{
		if cap(dest.mask)<src.DataSize(){
			dest.mask=make([]bool, src.DataSize())
		}
		dest.mask=dest.mask[:dest.DataSize()]
	}

	dt := dest.t
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next() ; err != nil {
			if err = handleNoOp(err); err != nil{
				return count, err
			}
			break
		}
		if j, err = siter.Next() ; err != nil {
			if err = handleNoOp(err); err != nil{
				return count, err
			}
			break
		}
		if isMasked{
			dest.mask[i]=src.mask[j]
		}
		
		switch dt {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case {{reflectKind .}}:
			dest.{{setOne .}}(i, src.{{getOne .}}(j))
			{{end -}}
		{{end -}}
		default:
			dest.Set(i, src.Get(j))
		}
		count++
	}
	return count, err
}
`

const sliceRaw = `// the method assumes the AP and metadata has already been set and this is simply slicing the values
func (t *Dense) slice(start, end int) {
	switch t.t {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case {{reflectKind .}}:
		data := t.{{sliceOf .}}[start:end]
		t.fromSlice(data)
		{{end -}}
	{{end -}}
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}	
}
`

var (
	CopySliced *template.Template
	CopyIter   *template.Template
	Slice      *template.Template
)

func init() {

	CopySliced = template.Must(template.New("copySliced").Funcs(funcs).Parse(copySlicedRaw))
	CopyIter = template.Must(template.New("copyIter").Funcs(funcs).Parse(copyIterRaw))
	Slice = template.Must(template.New("slice").Funcs(funcs).Parse(sliceRaw))
}

func GenerateDenseGetSet(f io.Writer, generic Kinds) {

	// CopySliced.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
	// CopyIter.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
	// Slice.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
}
