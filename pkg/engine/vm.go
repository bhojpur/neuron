package engine

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
	"bytes"
	"log"

	"github.com/bhojpur/neuron/pkg/tensor"
)

// VM represents a structure that can execute a graph or program. There are two VMs (both unexported):
//		- *tapeMachine
//		- *lispMachine
//
// The *tapeMachine pre-compiles a graph into a list of instructions, then executes the instructions linearly and sequentially.
// The main tradeoff is dynamism. Graphs cannot be dynamically created on the fly as a re-compilation process is required
// (and compilation is relatively expensive). However, graphs executed with the *tapeMachine run much faster as plenty of optimizations
// has been done in the code generation stage.
//
// The *lispMachine allows for graphs to be dynamically built and executed upon. The tradeoff is that executing a graph on *lispMachine
// is generally slower than on *tapeMachine, given the same static "image" of a graph.
type VM interface {
	RunAll() error
	Reset()

	// Close closes all the machine resources (CUDA, if any, loggers if any)
	Close() error
}

const (
	fwdOnly byte = iota
	bwdOnly
	watchNaN
	watchInf
	watchPointer
	allocVals
	spare2 // spare2 = trace in tapeVM,
	spare3 // spare3 = bindDV in tapeVM, manualRootGrad in LispVM
	watchAll
)

// VMOpt is a VM creation option
type VMOpt func(m VM)

// EvalMode enables the eval mode for the VM and graph
func EvalMode() VMOpt {
	return func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.evalMode = true
		case *tapeMachine:
			v.evalMode = true
		default:
			panic(nyi("EvalMode", v))
		}
	}
}

// WithLogger creates a VM with the supplied logger.
func WithLogger(logger *log.Logger) VMOpt {
	f := func(m VM) {
		if logger == nil {
			return
		}
		switch v := m.(type) {
		case *lispMachine:
			v.logger = logger
			v.buf = new(bytes.Buffer)
		case *tapeMachine:
			v.logger = logger
			v.buf = new(bytes.Buffer)
		default:
			panic(nyi("WithLogger", v))
		}
	}
	return f
}

// WithValueFmt defines how the logger will output the values. It defaults to "%3.3f"
func WithValueFmt(format string) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.valueFmt = format
		case *tapeMachine:
			v.valueFmt = format
		default:
			panic(nyi("WithValueFmt", v))
		}
	}
	return f
}

// WithWatchlist creates a VM with a watchlist. When the execution touches the things in the watchlist, the VM's logger will the log it.
// This allows for watching and finetuning of the algorithm. When nothing is passed in, then the VM will default to watching and logging every single
// execution object.
//
// The watchlist allows for different things to be watched, depending on VM type:
//		*lispMachine will ONLY take *Node
//		*tapeMachine will take int (for register IDs) or *Node.
func WithWatchlist(list ...interface{}) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			if len(list) == 0 {
				v.doWatchAll()
				return
			}

			for _, item := range list {
				n := item.(*Node) // will panic if node is not passed in. This is expected behaviour.
				v.watchlist = append(v.watchlist, n)
			}
		case *tapeMachine:
			if len(list) == 0 {
				v.doWatchAll()
				return
			}

			for _, item := range list {
				switch i := item.(type) {
				case int:
					v.watchRegs = append(v.watchRegs, register{id: i})
				case NodeID:
					v.watchNodeIDs = append(v.watchNodeIDs, i)
				case *Node:
					v.watchNodes = append(v.watchNodes, i)
				default:
					panic("WithWatchlist only works with register ids or nodes")
				}
			}
		default:
			panic(nyi("WithWatchlist", v))
		}
	}
	return f
}

// WithNaNWatch creates a VM that will watch for NaNs when executing. This slows the execution down.
func WithNaNWatch() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doWatchNaN()
		case *tapeMachine:
			v.doWatchNaN()
		default:
			panic(nyi("withNaNWatch", v))
		}
	}
	return f
}

// WithInfWatch creates a VM that will watch for Infs when executing. It watches for +Inf, -Inf and Inf. No choice there. This slows the execution down.
func WithInfWatch() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doWatchInf()
		case *tapeMachine:
			v.doWatchInf()
		default:
			panic(nyi("withInfWatch", v))
		}
	}
	return f
}

// WithPointerWatch creates a VM that will watch for pointer clashes when executing. This
// slows the execution down and it's only recommended for Bhojpur Neuron development.
func WithPointerWatch() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			panic("pointer watch not supported by the Lisp Machine yet")
		case *tapeMachine:
			v.doWatchPointer()
		default:
			panic(nyi("withInfWatch", v))
		}
	}
	return f
}

// ExecuteFwdOnly creates a VM that will execute a graph forwards only - it will not do back propagation.
// This option is only for *lispMachine. Try it on any other VMs and it will panic.
func ExecuteFwdOnly() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doExecFwd()
			v.dontExecBwd()
		default:
			panic(nyi("ExecuteFwdOnly", v))
		}
	}
	return f
}

// ExecuteBwdOnly creates a VM that will execute a graph by doing back propagation only.
// The assumption is of course, that the forward graph has already been executed, and there
// are already values associated with the nodes.
// This option is only for *lispMachine. Try it on any other VMs and it will panic.
func ExecuteBwdOnly() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doExecBwd()
			v.dontExecFwd()
		default:
			panic(nyi("ExecuteBwdOnly", v))
		}
	}
	return f
}

// LogFwd logs the forward execution of a graph.
// This option is only for *lispMachine. Try it on any other VMs and it will panic.
func LogFwd() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doLogFwd()
		default:
			panic(nyi("LogFwdOnly", v))
		}
	}
	return f
}

// LogBwd logs the backwards execution of a graph.
// This option is only for *lispMachine. Try it on any other VMs and it will panic.
func LogBwd() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doLogBwd()
		default:
			panic(nyi("LogBwdOnly", v))
		}
	}
	return f
}

// LogBothDir logs both directions of the execution of the graph.
// This option is only available for *lispMachine.
func LogBothDir() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.doLogFwd()
			v.doLogBwd()
		default:
			panic(nyi("LogBothDir", v))
		}
	}
	return f
}

// TraceExec is an option for *tapeMachine only.
// It stores an immutable copy of the executed value into the node, instead of a mutable value, which may be clobbered
func TraceExec() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *tapeMachine:
			v.doTrace()
		default:
			panic(nyi("TraceExec", v))
		}
	}
	return f
}

// BindDualValues is an option for *tapeMachine only.
// This is useful to set when using a Solver
func BindDualValues(nodes ...*Node) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *tapeMachine:
			v.doBindDV()
			v.bindNodesDV = append(v.bindNodesDV, nodes...)
			v.bindNodesDV = v.bindNodesDV.Set()
		default:
			// on by default for LispMachine
		}
	}
	return f
}

// WithPrecompiled is an option to pass in compiled programs.
// This is useful for users who use the CompileFunction function
func WithPrecompiled(prog *program, locMap map[*Node]register) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *tapeMachine:
			v.p = prog
			v.locMap = locMap
			v.cpumem = make([]Value, prog.cpulocs)
			v.gpumem = make([]Value, prog.gpulocs)
		default:
			// no op
		}
	}
	return f
}

// WithManualGradient allows the user to set the gradient of the root, before backprop. The root gradients should be set using the SetDeriv method
func WithManualGradient() VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.allowSetRootGrad()
		default:
			// noop
		}
	}
	return f
}

// WithEngine sets the tensor engine for computation inside the VM.
func WithEngine(e tensor.Engine) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *lispMachine:
			v.setEngine(e)
		case *tapeMachine:
			v.setEngine(e)
		}
	}
	return f
}
