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

// Subs is a list of substitution. Internally there are two very basic substitutions - one backed by map and the other a normal slice
type Subs interface {
	Get(TypeVariable) (Type, bool)
	Add(TypeVariable, Type) Subs
	Remove(TypeVariable) Subs

	// Iter() <-chan Substitution
	Iter() []Substitution
	Size() int
	Clone() Subs
}

// A Substitution is a tuple representing the TypeVariable and the replacement Type
type Substitution struct {
	Tv TypeVariable
	T  Type
}

type sSubs struct {
	s []Substitution
}

func newSliceSubs(maybeSize ...int) *sSubs {
	var size int
	if len(maybeSize) > 0 && maybeSize[0] > 0 {
		size = maybeSize[0]
	}
	retVal := BorrowSSubs(size)
	retVal.s = retVal.s[:0]
	return retVal
}

func (s *sSubs) Get(tv TypeVariable) (Type, bool) {
	if i := s.index(tv); i >= 0 {
		return s.s[i].T, true
	}
	return nil, false
}

func (s *sSubs) Add(tv TypeVariable, t Type) Subs {
	if i := s.index(tv); i >= 0 {
		s.s[i].T = t
		return s
	}
	s.s = append(s.s, Substitution{tv, t})
	return s
}

func (s *sSubs) Remove(tv TypeVariable) Subs {
	if i := s.index(tv); i >= 0 {
		// for now we keep the order
		copy(s.s[i:], s.s[i+1:])
		s.s[len(s.s)-1].T = nil
		s.s = s.s[:len(s.s)-1]
	}

	return s
}

func (s *sSubs) Iter() []Substitution { return s.s }
func (s *sSubs) Size() int            { return len(s.s) }
func (s *sSubs) Clone() Subs {
	retVal := BorrowSSubs(len(s.s))
	copy(retVal.s, s.s)
	return retVal
}

func (s *sSubs) index(tv TypeVariable) int {
	for i, sub := range s.s {
		if sub.Tv == tv {
			return i
		}
	}
	return -1
}

func (s *sSubs) Format(state fmt.State, c rune) {
	state.Write([]byte{'{'})
	for i, v := range s.s {
		if i < len(s.s)-1 {
			fmt.Fprintf(state, "%v: %v, ", v.Tv, v.T)

		} else {
			fmt.Fprintf(state, "%v: %v", v.Tv, v.T)
		}
	}
	state.Write([]byte{'}'})
}

type mSubs map[TypeVariable]Type

func (s mSubs) Get(tv TypeVariable) (Type, bool) { retVal, ok := s[tv]; return retVal, ok }
func (s mSubs) Add(tv TypeVariable, t Type) Subs { s[tv] = t; return s }
func (s mSubs) Remove(tv TypeVariable) Subs      { delete(s, tv); return s }

func (s mSubs) Iter() []Substitution {
	retVal := make([]Substitution, len(s))
	var i int
	for k, v := range s {
		retVal[i] = Substitution{k, v}
		i++
	}
	return retVal
}

func (s mSubs) Size() int { return len(s) }
func (s mSubs) Clone() Subs {
	retVal := make(mSubs)
	for k, v := range s {
		retVal[k] = v
	}
	return retVal
}

func compose(a, b Subs) (retVal Subs) {
	if b == nil {
		return a
	}

	retVal = b.Clone()

	if a == nil {
		return
	}

	for _, v := range a.Iter() {
		retVal = retVal.Add(v.Tv, v.T)
	}

	for _, v := range retVal.Iter() {
		retVal = retVal.Add(v.Tv, v.T.Apply(a).(Type))
	}
	return retVal
}
