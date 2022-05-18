//go:build instrumentation && !debug
// +build instrumentation,!debug

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

import (
	"log"
	"runtime"
	"sync"
)

const DEBUG = false

var tc uint32

// var _logger_ = log.New(os.Stderr, "", 0)
// var replacement = "\n"

func tabcount() int                             { return 0 }
func enterLoggingContext()                      {}
func leaveLoggingContext()                      {}
func logf(format string, others ...interface{}) {}

func logCaller(inspect string) {
	pc, _, _, _ := runtime.Caller(2)
	logf("%q Called by %v", inspect, runtime.FuncForPC(pc).Name())
}

/* Operational statistics related debugging */

var ql = new(sync.Mutex)
var q = make([]int, 0, 1000) // 1000 calls to DoWork
var blockingCallers = make(map[string]int)

func addQueueLength(l int) {
	ql.Lock()
	q = append(q, l)
	ql.Unlock()
}

// QueueLengths return the queue lengths recorded
func QueueLengths() []int {
	return q
}

// AverageQueueLength returns the average queue length recorded. This allows for optimizations.
func AverageQueueLength() int {
	ql.Lock()
	var s int
	for _, l := range q {
		s += l
	}
	avg := s / len(q) // yes, it's an integer division
	ql.Unlock()
	return avg
}

func addBlockingCallers() {
	pc, _, _, _ := runtime.Caller(3)
	fn := runtime.FuncForPC(pc)
	ql.Lock()
	blockingCallers[fn.Name()]++
	ql.Unlock()
}

func BlockingCallers() map[string]int {
	return blockingCallers
}

func (ctx *BatchedContext) QUEUE() []call {
	log.Println(len(ctx.queue))
	return ctx.queue
}

func (ctx *BatchedContext) Introspect() string {
	return ctx.introspect()
}
