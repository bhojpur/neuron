package hm

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

// A Namer is anything that knows its own name
type Namer interface {
	Name() string
}

// A Typer is an Expression node that knows its own Type
type Typer interface {
	Type() Type
}

// An Inferer is an Expression that can infer its own Type given an Env
type Inferer interface {
	Infer(Env, Fresher) (Type, error)
}

// An Expression is basically an AST node. In its simplest form, it's lambda calculus
type Expression interface {
	Body() Expression
}

// Var is an expression representing a variable
type Var interface {
	Expression
	Namer
	Typer
}

// Literal is an Expression/AST Node representing a literal
type Literal interface {
	Var
	IsLit() bool
}

// Apply is an Expression/AST node that represents a function application
type Apply interface {
	Expression
	Fn() Expression
}

// LetRec is an Expression/AST node that represents a recursive let
type LetRec interface {
	Let
	IsRecursive() bool
}

// Let is an Expression/AST node that represents the standard let polymorphism found in functional languages
type Let interface {
	// let name = def in body
	Expression
	Namer
	Def() Expression
}

// Lambda is an Expression/AST node that represents a function definiton
type Lambda interface {
	Expression
	Namer
	IsLambda() bool
}
