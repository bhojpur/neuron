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

// A Constraint is well. a constraint that says a must equal to b.
// It's used mainly in the constraint generation process.
type Constraint struct {
	a, b Type
}

func (c Constraint) Apply(sub Subs) Substitutable {
	c.a = c.a.Apply(sub).(Type)
	c.b = c.b.Apply(sub).(Type)
	return c
}

func (c Constraint) FreeTypeVar() TypeVarSet {
	var retVal TypeVarSet
	retVal = c.a.FreeTypeVar().Union(retVal)
	retVal = c.b.FreeTypeVar().Union(retVal)
	return retVal
}

func (c Constraint) Format(state fmt.State, r rune) {
	fmt.Fprintf(state, "{%v = %v}", c.a, c.b)
}
