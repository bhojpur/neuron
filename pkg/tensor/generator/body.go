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

import "text/template"

// generic loop templates

type LoopBody struct {
	TypedOp
	Range string
	Left  string
	Right string

	Index0, Index1, Index2 string

	IterName0, IterName1, IterName2 string
}

const (
	genericLoopRaw = `for i := range {{.Range}} {
		{{template "check" . -}}
		{{template "loopbody" . -}}
	}`

	genericUnaryIterLoopRaw = `var {{.Index0}} int
	var valid{{.Index0}} bool
	for {
		if {{.Index0}}, valid{{.Index0}}, err = {{.IterName0}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if valid{{.Index0}} {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	genericBinaryIterLoopRaw = `var {{.Index0}}, {{.Index1}} int
	var valid{{.Index0}}, valid{{.Index1}} bool
	for {
		if {{.Index0}}, valid{{.Index0}}, err = {{.IterName0}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if {{.Index1}}, valid{{.Index1}}, err = {{.IterName1}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if valid{{.Index0}} && valid{{.Index1}} {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	genericTernaryIterLoopRaw = `var {{.Index0}}, {{.Index1}}, {{.Index2}} int
	var valid{{.Index0}}, valid{{.Index1}}, valid{{.Index2}} bool
	for {
		if {{.Index0}}, valid{{.Index0}}, err = {{.IterName0}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if {{.Index1}}, valid{{.Index1}}, err = {{.IterName1}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if {{.Index2}}, valid{{.Index2}}, err = {{.IterName2}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if valid{{.Index0}} && valid{{.Index1}} && valid{{.Index2}} {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	// ALL THE SYNTACTIC ABSTRACTIONS!
	// did I mention how much I hate C-style macros? Now I'm doing them instead

	basicSet = `{{if .IsFunc -}}
			{{.Range}}[i] = {{ template "callFunc" . -}}
		{{else -}}
			{{.Range}}[i] = {{template "opDo" . -}}
		{{end -}}`

	basicIncr = `{{if .IsFunc -}}
			{{.Range}}[i] += {{template "callFunc" . -}}
		{{else -}}
			{{.Range}}[i] += {{template "opDo" . -}}
		{{end -}}`

	iterIncrLoopBody = `{{if .IsFunc -}}
			{{.Range}}[k] += {{template "callFunc" . -}}
		{{else -}}
			{{.Range}}[k] += {{template "opDo" . -}}
		{{end -}}`

	sameSet = `if {{template "opDo" . }} {
		{{.Range}}[i] = {{trueValue .Kind}}
	}else{
		{{.Range}}[i] = {{falseValue .Kind}}
	}`

	clampBody = `if {{.Range}}[i] < min {{if eq .Kind.String "float64"}}|| math.IsInf({{.Range}}[i], -1){{else if eq .Kind.String "float32"}}|| math32.IsInf({{.Range}}[i], -1){{end}}  {
		{{.Range}}[i] = min
		continue
	}
	if {{.Range}}[i] > max {{if eq .Kind.String "float64"}}|| math.IsInf({{.Range}}[i], 1){{else if eq .Kind.String "float32"}}|| math32.IsInf({{.Range}}[i], 1){{end}} {
		{{.Range}}[i] = max
	}
	`

	ternaryIterSet = `{{.Range}}[k] = {{template "opDo" . -}}`

	binOpCallFunc = `{{if eq "complex64" .Kind.String -}}
		complex64({{template "symbol" .Kind}}(complex128({{.Left}}), complex128({{.Right}}))){{else -}}
		{{template "symbol" .Kind}}({{.Left}}, {{.Right}}){{end -}}`

	binOpDo = `{{.Left}} {{template "symbol" .Kind}} {{.Right}}`

	unaryOpDo = `{{template "symbol" .}}{{.Left}}[{{.Index0}}]`

	unaryOpCallFunc = `{{if eq "complex64" .Kind.String -}}
		complex64({{template "symbol" .}}(complex128({{.Left}}[{{.Index0}}]))){{else -}}
		{{template "symbol" .}}({{.Left}}[{{.Index0}}]){{end -}}
		`

	check0 = `if {{.Right}} == 0 {
		errs = append(errs, i)
		{{.Range}}[i] = 0
		continue
	}
	`

	maskCheck = `if mask[i] {
		continue
	}
	`

	genericArgmaxVarDeclRaw = `var set bool
	var f {{asType .Kind}}
	var {{.ArgX | lower}} int
	`
)

// renamed
const (
	vvLoopRaw         = genericLoopRaw
	vvIncrLoopRaw     = genericLoopRaw
	vvIterLoopRaw     = genericBinaryIterLoopRaw
	vvIterIncrLoopRaw = genericTernaryIterLoopRaw

	mixedLoopRaw         = genericLoopRaw
	mixedIncrLoopRaw     = genericLoopRaw
	mixedIterLoopRaw     = genericUnaryIterLoopRaw
	mixedIterIncrLoopRaw = genericBinaryIterLoopRaw
)

var genericArgmaxVarDecl *template.Template

func init() {
	genericArgmaxVarDecl = template.Must(template.New("genericArgmaxVarDecl").Funcs(funcs).Parse(genericArgmaxVarDeclRaw))
}
