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
	"sort"

	"github.com/xtgo/set"
)

// TypeVarSet is a set of TypeVariable
type TypeVarSet []TypeVariable

// TypeVariables are orderable, so we fulfil the interface for sort.Interface

func (s TypeVarSet) Len() int           { return len(s) }
func (s TypeVarSet) Less(i, j int) bool { return s[i] < s[j] }
func (s TypeVarSet) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func (s TypeVarSet) Set() TypeVarSet {
	sort.Sort(s)
	n := set.Uniq(s)
	s = s[:n]
	return s
}

func (s TypeVarSet) Union(other TypeVarSet) TypeVarSet {
	if other == nil {
		return s
	}

	sort.Sort(s)
	sort.Sort(other)
	s2 := append(s, other...)
	n := set.Union(s2, len(s))
	return s2[:n]
}

func (s TypeVarSet) Intersect(other TypeVarSet) TypeVarSet {
	if len(s) == 0 || len(other) == 0 {
		return nil
	}

	sort.Sort(s)
	sort.Sort(other)
	s2 := append(s, other...)
	n := set.Inter(s2, len(s))
	return s2[:n]
}

func (s TypeVarSet) Difference(other TypeVarSet) TypeVarSet {
	sort.Sort(s)
	sort.Sort(other)
	s2 := append(s, other...)
	n := set.Diff(s2, len(s))
	return s2[:n]
}

func (s TypeVarSet) Contains(tv TypeVariable) bool {
	for _, v := range s {
		if v == tv {
			return true
		}
	}
	return false
}

func (s TypeVarSet) Index(tv TypeVariable) int {
	for i, v := range s {
		if v == tv {
			return i
		}
	}
	return -1
}

func (s TypeVarSet) Equals(other TypeVarSet) bool {
	if len(s) != len(other) {
		return false
	}

	if len(s) == 0 {
		return true
	}

	if &s[0] == &other[0] {
		return true
	}

	for _, v := range s {
		if !other.Contains(v) {
			return false
		}
	}
	return true
}
