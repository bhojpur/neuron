//go:build !debug && !instrumentation
// +build !debug,!instrumentation

package cu

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

const DEBUG = false

var tc uint32

// var _logger_ = log.New(os.Stderr, "", 0)
// var replacement = "\n"

func tabcount() int                             { return 0 }
func enterLoggingContext()                      {}
func leaveLoggingContext()                      {}
func logf(format string, others ...interface{}) {}

/* Debugging Utility Methods */

// instrumentation related functions
func logCaller(inspect string) {}

func addQueueLength(l int) {}

// QueueLengths return the queue lengths recorded
func QueueLengths() []int { return nil }

// AverageQueueLength returns the average queue length recorded. This allows for optimizations.
func AverageQueueLength() int { return 0 }

func addBlockingCallers() {}

func BlockingCallers() map[string]int { return nil }
