package main

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
	"text/template"

	bg "github.com/bhojpur/neuron/pkg/bindgen"
	"github.com/cznic/cc"
)

var skip = map[string]bool{
	"cublasErrprn":    true,
	"cublasSrotg":     true,
	"cublasSrotmg":    true,
	"cublasSrotm":     true,
	"cublasDrotg":     true,
	"cublasDrotmg":    true,
	"cublasDrotm":     true,
	"cublasCrotg":     true,
	"cublasZrotg":     true,
	"cublasCdotu_sub": true,
	"cublasCdotc_sub": true,
	"cublasZdotu_sub": true,
	"cublasZdotc_sub": true,

	// ATLAS extensions.
	"cublasCsrot": true,
	"cublasZdrot": true,

	// trmm
	"cublasStrmm": true,
	"cublasDtrmm": true,
	"cublasZtrmm": true,
	"cublasCtrmm": true,
}

var cToGoType = map[string]string{
	"int":    "int",
	"float":  "float32",
	"double": "float64",
}

var blasEnums = map[string]bg.Template{
	"CUBLAS_ORDER":     bg.Pure(template.Must(template.New("order").Parse("order"))),
	"CUBLAS_DIAG":      bg.Pure(template.Must(template.New("diag").Parse("blas.Diag"))),
	"CUBLAS_TRANSPOSE": bg.Pure(template.Must(template.New("trans").Parse("blas.Transpose"))),
	"CUBLAS_UPLO":      bg.Pure(template.Must(template.New("uplo").Parse("blas.Uplo"))),
	"CUBLAS_SIDE":      bg.Pure(template.Must(template.New("side").Parse("blas.Side"))),
}

var cgoEnums = map[string]bg.Template{
	"CUBLAS_ORDER":     bg.Pure(template.Must(template.New("order").Parse("C.enum_CBLAS_ORDER(rowMajor)"))),
	"CUBLAS_DIAG":      bg.Pure(template.Must(template.New("diag").Parse("diag2cublasDiag({{.}})"))),
	"CUBLAS_TRANSPOSE": bg.Pure(template.Must(template.New("trans").Parse("trans2cublasTrans({{.}})"))),
	"CUBLAS_UPLO":      bg.Pure(template.Must(template.New("uplo").Parse("uplo2cublasUplo({{.}})"))),
	"CUBLAS_SIDE":      bg.Pure(template.Must(template.New("side").Parse("side2cublasSide({{.}})"))),
}

var (
	complex64Type = map[bg.TypeKey]bg.Template{
		{Kind: cc.FloatComplex, IsPointer: true}: bg.Pure(template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex64{{else}}[]complex64{{end}}`,
		)))}

	complex128Type = map[bg.TypeKey]bg.Template{
		{Kind: cc.DoubleComplex, IsPointer: true}: bg.Pure(template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex128{{else}}[]complex128{{end}}`,
		)))}
)

var names = map[string]string{
	"uplo":   "ul",
	"trans":  "t",
	"transA": "tA",
	"transB": "tB",
	"side":   "s",
	"diag":   "d",
}
