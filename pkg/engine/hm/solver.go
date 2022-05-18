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

type solver struct {
	sub Subs
	err error
}

func newSolver() *solver {
	return new(solver)
}

func (s *solver) solve(cs Constraints) {
	logf("solving constraints: %d", len(cs))
	enterLoggingContext()
	defer leaveLoggingContext()
	logf("starting sub %v", s.sub)
	if s.err != nil {
		return
	}

	switch len(cs) {
	case 0:
		return
	default:
		var sub Subs
		c := cs[0]
		sub, s.err = Unify(c.a, c.b)
		defer ReturnSubs(s.sub)

		s.sub = compose(sub, s.sub)
		cs = cs[1:].Apply(s.sub).(Constraints)
		s.solve(cs)

	}
	logf("Ending Sub %v", s.sub)
	return
}
