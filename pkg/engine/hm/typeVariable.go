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

import (
	"fmt"

	"github.com/pkg/errors"
)

// TypeVariable is a variable that ranges over the types - that is to say it can take any type.
type TypeVariable rune

func (t TypeVariable) Name() string { return string(t) }
func (t TypeVariable) Apply(sub Subs) Substitutable {
	if sub == nil {
		return t
	}

	if retVal, ok := sub.Get(t); ok {
		return retVal
	}

	return t
}

func (t TypeVariable) FreeTypeVar() TypeVarSet { tvs := BorrowTypeVarSet(1); tvs[0] = t; return tvs }
func (t TypeVariable) Normalize(k, v TypeVarSet) (Type, error) {
	if i := k.Index(t); i >= 0 {
		return v[i], nil
	}
	return nil, errors.Errorf("Type Variable %v not in signature", t)
}

func (t TypeVariable) Types() Types               { return nil }
func (t TypeVariable) String() string             { return string(t) }
func (t TypeVariable) Format(s fmt.State, c rune) { fmt.Fprintf(s, "%c", rune(t)) }
func (t TypeVariable) Eq(other Type) bool         { return other == t }
