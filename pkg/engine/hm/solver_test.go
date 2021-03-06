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

import "testing"

var solverTest = []struct {
	cs Constraints

	expected Subs
	err      bool
}{
	{Constraints{{TypeVariable('a'), proton}}, mSubs{'a': proton}, false},
	{Constraints{{NewFnType(TypeVariable('a'), proton), neutron}}, nil, true},
	{Constraints{{NewFnType(TypeVariable('a'), proton), NewFnType(proton, proton)}}, mSubs{'a': proton}, false},

	{Constraints{
		{
			NewFnType(TypeVariable('a'), TypeVariable('a'), list{TypeVariable('a')}),
			NewFnType(proton, proton, TypeVariable('b')),
		},
	},
		mSubs{'a': proton, 'b': list{proton}}, false,
	},

	{
		Constraints{
			{TypeVariable('a'), TypeVariable('b')},
			{TypeVariable('a'), proton},
		},
		mSubs{'a': proton}, false,
	},

	{
		Constraints{
			{
				NewRecordType("", TypeVariable('a'), TypeVariable('a'), TypeVariable('b')),
				NewRecordType("", neutron, neutron, proton),
			},
		},
		mSubs{'a': neutron, 'b': proton}, false,
	},
}

func TestSolver(t *testing.T) {
	for i, sts := range solverTest {
		solver := newSolver()
		solver.solve(sts.cs)

		if sts.err {
			if solver.err == nil {
				t.Errorf("Test %d Expected an error", i)
			}
			continue
		} else if solver.err != nil {
			t.Error(solver.err)
		}

		for _, v := range sts.expected.Iter() {
			if T, ok := solver.sub.Get(v.Tv); !ok {
				t.Errorf("Test %d: Expected type variable %v in subs: %v", i, v.Tv, solver.sub)
				break
			} else if T != v.T {
				t.Errorf("Test %d: Expected replacement to be %v. Got %v instead", i, v.T, T)
			}
		}
	}
}
