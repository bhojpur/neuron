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

import "sync"

const (
	poolSize = 4
	extraCap = 2
)

var sSubPool = [poolSize]*sync.Pool{
	&sync.Pool{
		New: func() interface{} { return &sSubs{s: make([]Substitution, 1, 1+extraCap)} },
	},
	&sync.Pool{
		New: func() interface{} { return &sSubs{s: make([]Substitution, 2, 2+extraCap)} },
	},
	&sync.Pool{
		New: func() interface{} { return &sSubs{s: make([]Substitution, 3, 3+extraCap)} },
	},
	&sync.Pool{
		New: func() interface{} { return &sSubs{s: make([]Substitution, 4, 4+extraCap)} },
	},
}

var mSubPool = &sync.Pool{
	New: func() interface{} { return make(mSubs) },
}

// ReturnSubs returns substitutions to the pool. USE WITH CAUTION.
func ReturnSubs(sub Subs) {
	switch s := sub.(type) {
	case mSubs:
		for k := range s {
			delete(s, k)
		}
		mSubPool.Put(sub)
	case *sSubs:
		size := cap(s.s) - 2
		if size > 0 && size < poolSize+1 {
			// reset to empty
			for i := range s.s {
				s.s[i] = Substitution{}
			}

			s.s = s.s[:size]
			sSubPool[size-1].Put(sub)
		}
	}
}

// BorrowMSubs gets a map based substitution from a shared pool. USE WITH CAUTION
func BorrowMSubs() mSubs {
	return mSubPool.Get().(mSubs)
}

// BorrowSSubs gets a slice based substituiton from a shared pool. USE WITH CAUTION
func BorrowSSubs(size int) *sSubs {
	if size > 0 && size < 5 {
		retVal := sSubPool[size-1].Get().(*sSubs)
		return retVal
	}
	s := make([]Substitution, size)
	return &sSubs{s: s}
}

var typesPool = [poolSize]*sync.Pool{
	&sync.Pool{
		New: func() interface{} { return make(Types, 1) },
	},

	&sync.Pool{
		New: func() interface{} { return make(Types, 2) },
	},

	&sync.Pool{
		New: func() interface{} { return make(Types, 3) },
	},

	&sync.Pool{
		New: func() interface{} { return make(Types, 4) },
	},
}

// BorrowTypes gets a slice of Types with size. USE WITH CAUTION.
func BorrowTypes(size int) Types {
	if size > 0 && size < poolSize+1 {
		return typesPool[size-1].Get().(Types)
	}
	return make(Types, size)
}

// ReturnTypes returns the slice of types into the pool. USE WITH CAUTION
func ReturnTypes(ts Types) {
	if size := cap(ts); size > 0 && size < poolSize+1 {
		ts = ts[:cap(ts)]
		for i := range ts {
			ts[i] = nil
		}
		typesPool[size-1].Put(ts)
	}
}

var typeVarSetPool = [poolSize]*sync.Pool{
	&sync.Pool{
		New: func() interface{} { return make(TypeVarSet, 1) },
	},

	&sync.Pool{
		New: func() interface{} { return make(TypeVarSet, 2) },
	},

	&sync.Pool{
		New: func() interface{} { return make(TypeVarSet, 3) },
	},

	&sync.Pool{
		New: func() interface{} { return make(TypeVarSet, 4) },
	},
}

// BorrowTypeVarSet gets a TypeVarSet of size from pool. USE WITH CAUTION
func BorrowTypeVarSet(size int) TypeVarSet {
	if size > 0 && size < poolSize+1 {
		return typeVarSetPool[size-1].Get().(TypeVarSet)
	}
	return make(TypeVarSet, size)
}

// ReturnTypeVarSet returns the TypeVarSet to pool. USE WITH CAUTION
func ReturnTypeVarSet(ts TypeVarSet) {
	var def TypeVariable
	if size := cap(ts); size > 0 && size < poolSize+1 {
		ts = ts[:cap(ts)]
		for i := range ts {
			ts[i] = def
		}
		typeVarSetPool[size-1].Put(ts)
	}
}

var fnTypePool = &sync.Pool{
	New: func() interface{} { return new(FunctionType) },
}

func borrowFnType() *FunctionType {
	return fnTypePool.Get().(*FunctionType)
}

// ReturnFnType returns a *FunctionType to the pool. NewFnType automatically borrows from the pool. USE WITH CAUTION
func ReturnFnType(fnt *FunctionType) {
	if a, ok := fnt.a.(*FunctionType); ok {
		ReturnFnType(a)
	}

	if b, ok := fnt.b.(*FunctionType); ok {
		ReturnFnType(b)
	}

	fnt.a = nil
	fnt.b = nil
	fnTypePool.Put(fnt)
}
