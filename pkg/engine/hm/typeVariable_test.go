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
	"testing"
)

func TestTypeVariableBasics(t *testing.T) {
	tv := TypeVariable('a')
	if name := tv.Name(); name != "a" {
		t.Errorf("Expected name to be \"a\". Got %q instead", name)
	}

	if str := tv.String(); str != "a" {
		t.Errorf("Expected String() of 'a'. Got %q instead", str)
	}

	if tv.Types() != nil {
		t.Errorf("Expected Types() of TypeVariable to be nil")
	}

	ftv := tv.FreeTypeVar()
	if len(ftv) != 1 {
		t.Errorf("Expected a type variable to be free when FreeTypeVar() is called")
	}

	if ftv[0] != tv {
		t.Errorf("Expected ...")
	}

	sub := mSubs{
		'a': proton,
	}

	if tv.Apply(sub) != proton {
		t.Error("Expected proton")
	}

	sub = mSubs{
		'b': proton,
	}

	if tv.Apply(sub) != tv {
		t.Error("Expected unchanged")
	}
}

func TestTypeVariableNormalize(t *testing.T) {
	original := TypeVarSet{'c', 'a', 'd'}
	normalized := TypeVarSet{'a', 'b', 'c'}

	tv := TypeVariable('a')
	norm, err := tv.Normalize(original, normalized)
	if err != nil {
		t.Error(err)
	}

	if norm != TypeVariable('b') {
		t.Errorf("Expected 'b'. Got %v", norm)
	}

	tv = TypeVariable('e')
	if _, err = tv.Normalize(original, normalized); err == nil {
		t.Error("Expected an error")
	}
}

func TestTypeConst(t *testing.T) {
	T := proton
	if T.Name() != "proton" {
		t.Error("Expected name to be proton")
	}

	if fmt.Sprintf("%v", T) != "proton" {
		t.Error("Expected name to be proton")
	}

	if T.String() != "proton" {
		t.Error("Expected name to be proton")
	}

	if T2, err := T.Normalize(nil, nil); err != nil {
		t.Error(err)
	} else if T2 != T {
		t.Error("Const types should return itself")
	}
}
