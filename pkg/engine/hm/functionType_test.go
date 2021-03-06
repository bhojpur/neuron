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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFunctionTypeBasics(t *testing.T) {
	fnType := NewFnType(TypeVariable('a'), TypeVariable('a'), TypeVariable('a'))
	if fnType.Name() != "→" {
		t.Errorf("FunctionType should have \"→\" as a name. Got %q instead", fnType.Name())
	}

	if fnType.String() != "a → a → a" {
		t.Errorf("Expected \"a → a → a\". Got %q instead", fnType.String())
	}

	if !fnType.Arg().Eq(TypeVariable('a')) {
		t.Error("Expected arg of function to be 'a'")
	}

	if !fnType.Ret(false).Eq(NewFnType(TypeVariable('a'), TypeVariable('a'))) {
		t.Error("Expected ret(false) to be a → a")
	}

	if !fnType.Ret(true).Eq(TypeVariable('a')) {
		t.Error("Expected final return type to be 'a'")
	}

	// a very simple fn
	fnType = NewFnType(TypeVariable('a'), TypeVariable('a'))
	if !fnType.Ret(true).Eq(TypeVariable('a')) {
		t.Error("Expected final return type to be 'a'")
	}

	ftv := fnType.FreeTypeVar()
	if len(ftv) != 1 {
		t.Errorf("Expected only one free type var")
	}

	for _, fas := range fnApplyTests {
		fn := fas.fn.Apply(fas.sub).(*FunctionType)
		if !fn.Eq(fas.expected) {
			t.Errorf("Expected %v. Got %v instead", fas.expected, fn)
		}
	}

	// bad shit
	f := func() {
		NewFnType(TypeVariable('a'))
	}
	assert.Panics(t, f)
}

var fnApplyTests = []struct {
	fn  *FunctionType
	sub Subs

	expected *FunctionType
}{
	{NewFnType(TypeVariable('a'), TypeVariable('a')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, proton)},
	{NewFnType(TypeVariable('a'), TypeVariable('b')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, neutron)},
	{NewFnType(TypeVariable('a'), TypeVariable('b')), mSubs{'c': proton, 'd': neutron}, NewFnType(TypeVariable('a'), TypeVariable('b'))},
	{NewFnType(TypeVariable('a'), TypeVariable('b')), mSubs{'a': proton, 'c': neutron}, NewFnType(proton, TypeVariable('b'))},
	{NewFnType(TypeVariable('a'), TypeVariable('b')), mSubs{'c': proton, 'b': neutron}, NewFnType(TypeVariable('a'), neutron)},
	{NewFnType(electron, proton), mSubs{'a': proton, 'b': neutron}, NewFnType(electron, proton)},

	// a -> (b -> c)
	{NewFnType(TypeVariable('a'), TypeVariable('b'), TypeVariable('a')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, neutron, proton)},
	{NewFnType(TypeVariable('a'), TypeVariable('a'), TypeVariable('b')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, proton, neutron)},
	{NewFnType(TypeVariable('a'), TypeVariable('b'), TypeVariable('c')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, neutron, TypeVariable('c'))},
	{NewFnType(TypeVariable('a'), TypeVariable('c'), TypeVariable('b')), mSubs{'a': proton, 'b': neutron}, NewFnType(proton, TypeVariable('c'), neutron)},

	// (a -> b) -> c
	{NewFnType(NewFnType(TypeVariable('a'), TypeVariable('b')), TypeVariable('a')), mSubs{'a': proton, 'b': neutron}, NewFnType(NewFnType(proton, neutron), proton)},
}

func TestFunctionType_FlatTypes(t *testing.T) {
	fnType := NewFnType(TypeVariable('a'), TypeVariable('b'), TypeVariable('c'))
	ts := fnType.FlatTypes()
	correct := Types{TypeVariable('a'), TypeVariable('b'), TypeVariable('c')}
	assert.Equal(t, ts, correct)

	fnType2 := NewFnType(fnType, TypeVariable('d'))
	correct = append(correct, TypeVariable('d'))
	ts = fnType2.FlatTypes()
	assert.Equal(t, ts, correct)
}

func TestFunctionType_Clone(t *testing.T) {
	fnType := NewFnType(TypeVariable('a'), TypeVariable('b'), TypeVariable('c'))
	assert.Equal(t, fnType.Clone(), fnType)

	rec := NewRecordType("", TypeVariable('a'), NewFnType(TypeVariable('a'), TypeVariable('b')), TypeVariable('c'))
	fnType = NewFnType(rec, rec)
	assert.Equal(t, fnType.Clone(), fnType)
}
