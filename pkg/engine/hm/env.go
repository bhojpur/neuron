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

// An Env is essentially a map of names to schemes
type Env interface {
	Substitutable
	SchemeOf(string) (*Scheme, bool)
	Clone() Env

	Add(string, *Scheme) Env
	Remove(string) Env
}

type SimpleEnv map[string]*Scheme

func (e SimpleEnv) Apply(sub Subs) Substitutable {
	logf("Applying %v to env", sub)
	if sub == nil {
		return e
	}

	for _, v := range e {
		v.Apply(sub) // apply mutates Scheme, so no need to set
	}
	return e
}

func (e SimpleEnv) FreeTypeVar() TypeVarSet {
	var retVal TypeVarSet
	for _, v := range e {
		retVal = v.FreeTypeVar().Union(retVal)
	}
	return retVal
}

func (e SimpleEnv) SchemeOf(name string) (retVal *Scheme, ok bool) { retVal, ok = e[name]; return }
func (e SimpleEnv) Clone() Env {
	retVal := make(SimpleEnv)
	for k, v := range e {
		retVal[k] = v.Clone()
	}
	return retVal
}

func (e SimpleEnv) Add(name string, s *Scheme) Env {
	e[name] = s
	return e
}

func (e SimpleEnv) Remove(name string) Env {
	delete(e, name)
	return e
}
