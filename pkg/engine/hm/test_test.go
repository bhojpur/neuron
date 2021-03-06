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

const (
	proton  TypeConst = "proton"
	neutron TypeConst = "neutron"
	quark   TypeConst = "quark"

	electron TypeConst = "electron"
	positron TypeConst = "positron"
	muon     TypeConst = "muon"

	photon TypeConst = "photon"
	higgs  TypeConst = "higgs"
)

type list struct {
	t Type
}

func (l list) Name() string                  { return "List" }
func (l list) Apply(subs Subs) Substitutable { l.t = l.t.Apply(subs).(Type); return l }
func (l list) FreeTypeVar() TypeVarSet       { return l.t.FreeTypeVar() }
func (l list) Format(s fmt.State, c rune)    { fmt.Fprintf(s, "List %v", l.t) }
func (l list) String() string                { return fmt.Sprintf("%v", l) }
func (l list) Normalize(k, v TypeVarSet) (Type, error) {
	var t Type
	var err error
	if t, err = l.t.Normalize(k, v); err != nil {
		return nil, err
	}
	l.t = t
	return l, nil
}
func (l list) Types() Types { return Types{l.t} }
func (l list) Eq(other Type) bool {
	if ot, ok := other.(list); ok {
		return ot.t.Eq(l.t)
	}
	return false
}

type mirrorUniverseList struct {
	t Type
}

func (l mirrorUniverseList) Name() string                  { return "GoateeList" }
func (l mirrorUniverseList) Apply(subs Subs) Substitutable { l.t = l.t.Apply(subs).(Type); return l }
func (l mirrorUniverseList) FreeTypeVar() TypeVarSet       { return l.t.FreeTypeVar() }
func (l mirrorUniverseList) Format(s fmt.State, c rune)    { fmt.Fprintf(s, "List %v", l.t) }
func (l mirrorUniverseList) String() string                { return fmt.Sprintf("%v", l) }
func (l mirrorUniverseList) Normalize(k, v TypeVarSet) (Type, error) {
	var t Type
	var err error
	if t, err = l.t.Normalize(k, v); err != nil {
		return nil, err
	}
	l.t = t
	return l, nil
}
func (l mirrorUniverseList) Types() Types { return Types{l.t} }
func (l mirrorUniverseList) Eq(other Type) bool {
	if ot, ok := other.(list); ok {
		return ot.t.Eq(l.t)
	}
	return false
}

// satisfies the Inferer interface for testing
type selfInferer bool

func (t selfInferer) Infer(Env, Fresher) (Type, error) {
	if bool(t) {
		return proton, nil
	}
	return nil, errors.Errorf("fail")
}
func (t selfInferer) Body() Expression { panic("not implemented") }

// satisfies the Var interface for testing. It also doesn't know its own type
type variable string

func (t variable) Body() Expression { return nil }
func (t variable) Name() string     { return string(t) }
func (t variable) Type() Type       { return nil }
