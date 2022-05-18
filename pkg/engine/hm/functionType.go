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

import "fmt"

// FunctionType is a type constructor that builds function types.
type FunctionType struct {
	a, b Type
}

// NewFnType creates a new FunctionType. Functions are by default right associative. This:
//		NewFnType(a, a, a)
// is short hand for this:
// 		NewFnType(a, NewFnType(a, a))
func NewFnType(ts ...Type) *FunctionType {
	if len(ts) < 2 {
		panic("Expected at least 2 input types")
	}

	retVal := borrowFnType()
	retVal.a = ts[0]

	if len(ts) > 2 {
		retVal.b = NewFnType(ts[1:]...)
	} else {
		retVal.b = ts[1]
	}
	return retVal
}

func (t *FunctionType) Name() string { return "→" }
func (t *FunctionType) Apply(sub Subs) Substitutable {
	t.a = t.a.Apply(sub).(Type)
	t.b = t.b.Apply(sub).(Type)
	return t
}

func (t *FunctionType) FreeTypeVar() TypeVarSet    { return t.a.FreeTypeVar().Union(t.b.FreeTypeVar()) }
func (t *FunctionType) Format(s fmt.State, c rune) { fmt.Fprintf(s, "%v → %v", t.a, t.b) }
func (t *FunctionType) String() string             { return fmt.Sprintf("%v", t) }
func (t *FunctionType) Normalize(k, v TypeVarSet) (Type, error) {
	var a, b Type
	var err error
	if a, err = t.a.Normalize(k, v); err != nil {
		return nil, err
	}

	if b, err = t.b.Normalize(k, v); err != nil {
		return nil, err
	}

	return NewFnType(a, b), nil
}
func (t *FunctionType) Types() Types {
	retVal := BorrowTypes(2)
	retVal[0] = t.a
	retVal[1] = t.b
	return retVal
}

func (t *FunctionType) Eq(other Type) bool {
	if ot, ok := other.(*FunctionType); ok {
		return ot.a.Eq(t.a) && ot.b.Eq(t.b)
	}
	return false
}

// Other methods (accessors mainly)

// Arg returns the type of the function argument
func (t *FunctionType) Arg() Type { return t.a }

// Ret returns the return type of a function. If recursive is true, it will get the final return type
func (t *FunctionType) Ret(recursive bool) Type {
	if !recursive {
		return t.b
	}

	if fnt, ok := t.b.(*FunctionType); ok {
		return fnt.Ret(recursive)
	}

	return t.b
}

// FlatTypes returns the types in FunctionTypes as a flat slice of types. This allows for easier iteration in some applications
func (t *FunctionType) FlatTypes() Types {
	retVal := BorrowTypes(8) // start with 8. Can always grow
	retVal = retVal[:0]

	if a, ok := t.a.(*FunctionType); ok {
		ft := a.FlatTypes()
		retVal = append(retVal, ft...)
		ReturnTypes(ft)
	} else {
		retVal = append(retVal, t.a)
	}

	if b, ok := t.b.(*FunctionType); ok {
		ft := b.FlatTypes()
		retVal = append(retVal, ft...)
		ReturnTypes(ft)
	} else {
		retVal = append(retVal, t.b)
	}
	return retVal
}

// Clone implements Cloner
func (t *FunctionType) Clone() interface{} {
	retVal := new(FunctionType)

	if ac, ok := t.a.(Cloner); ok {
		retVal.a = ac.Clone().(Type)
	} else {
		retVal.a = t.a
	}

	if bc, ok := t.b.(Cloner); ok {
		retVal.b = bc.Clone().(Type)
	} else {
		retVal.b = t.b
	}
	return retVal
}
