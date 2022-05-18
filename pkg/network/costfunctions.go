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
	"github.com/pkg/errors"
)

// RMS represents a root mean equare error
// The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently
// used measure of the differences between values (sample or population values) predicted
// by a model or an estimator and the values observed.
func RMS(yHat, y G.Input) (retVal *G.Node, err error) {
	if err = G.CheckOne(yHat); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from yHat")
	}

	if err = G.CheckOne(y); err != nil {
		return nil, errors.Wrap(err, "unable to extract node from y")
	}

	if retVal, err = G.Sub(yHat.Node(), y.Node()); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)")
	}

	if retVal, err = G.Square(retVal); err != nil {
		return nil, errors.Wrap(err, "(ŷ-y)²")
	}

	if retVal, err = G.Mean(retVal); err != nil {
		return nil, errors.Wrap(err, "mean((ŷ-y)²)")
	}

	return retVal, nil
}
