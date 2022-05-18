package network

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
	G "github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/engine/hm"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/pkg/errors"
)

type skip struct {
	b *G.Node
}

func ConsSkip(_ G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	l := &skip{}
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*skip); !ok {
			return nil, errors.Errorf("Construction option does not return *skip. Got %v of %T instead", o, o)
		}
	}
	return l, nil
}

func (l *skip) Model() G.Nodes { return nil }

func (l *skip) Fwd(x G.Input) G.Result {
	if err := G.CheckOne(x); err != nil {
		return G.Err(err)
	}
	return G.TransformResult(x, l.b)(G.Add(x.Node(), l.b))
}

func (l *skip) Name() string { return "+" + l.b.Name() }

func (l *skip) Type() hm.Type { return hm.NewFnType(l.b.Type(), l.b.Type()) }

func (l *skip) Shape() tensor.Shape { return l.b.Shape() }

func (l *skip) Describe() {}
