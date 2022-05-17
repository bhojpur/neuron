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

var arithDocStrings = map[string]*template.Template{
	"Add": template.Must(template.New("+").Parse("// Add performs {{.Left}} + {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"Sub": template.Must(template.New("-").Parse("// Sub performs {{.Left}} - {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"Mul": template.Must(template.New("×").Parse("// Mul performs {{.Left}} × {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"Div": template.Must(template.New("÷").Parse("// Div performs {{.Left}} ÷ {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"Pow": template.Must(template.New("^").Parse("// Pow performs {{.Left}} ^ {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"Mod": template.Must(template.New("%").Parse("// Mod performs {{.Left}} % {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),

	"AddScalar": template.Must(template.New("+").Parse("// AddScalar performs {{.Left}} + {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"SubScalar": template.Must(template.New("-").Parse("// SubScalar performs {{.Left}} - {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"MulScalar": template.Must(template.New("×").Parse("// MulScalar performs {{.Left}} × {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"DivScalar": template.Must(template.New("÷").Parse("// DivScalar performs {{.Left}} ÷ {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"PowScalar": template.Must(template.New("^").Parse("// PowScalar performs {{.Left}} ^ {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
	"ModScalar": template.Must(template.New("%").Parse("// ModScalar performs {{.Left}} % {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}.\n// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)\n")),
}

var cmpDocStrings = map[string]*template.Template{
	"Lt":   template.Must(template.New("+").Parse("// Lt performs {{.Left}} < {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"Lte":  template.Must(template.New("+").Parse("// Lte performs {{.Left}} ≤ {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"Gt":   template.Must(template.New("+").Parse("// Gt performs {{.Left}} > {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"Gte":  template.Must(template.New("+").Parse("// Gte performs {{.Left}} ≥ {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"ElEq": template.Must(template.New("+").Parse("// ElEq performs {{.Left}} == {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"ElNe": template.Must(template.New("+").Parse("// ElNe performs {{.Left}} ≠ {{.Right}} elementwise. Both {{.Left}} and {{.Right}} must have the same shape.\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n//UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),

	"LtScalar":   template.Must(template.New("+").Parse("// LtScalar performs {{.Left}} < {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"LteScalar":  template.Must(template.New("+").Parse("// LteScalar performs {{.Left}} ≤ {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"GtScalar":   template.Must(template.New("+").Parse("// GtScalar performs {{.Left}} > {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"GteScalar":  template.Must(template.New("+").Parse("// GteScalar performs {{.Left}} ≥ {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"ElEqScalar": template.Must(template.New("+").Parse("// EqScalar performs {{.Left}} == {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
	"ElNeScalar": template.Must(template.New("+").Parse("// NeScalar performs {{.Left}} ≠ {{.Right}} elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in {{.Right}}\n// Acceptable FuncOpts are: UseUnsafe(), AsSameType(), WithReuse().\n// UseUnsafe() will ensure that the same type is returned.\n// Tensors used in WithReuse has to have the same Dtype as the return value's Dtype.\n")),
}
