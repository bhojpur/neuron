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
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/topo"
)

// WalkGraph walks a graph. It returns a channel of *Nodes, so be sure to consume the channel or there may be a deadlock
func WalkGraph(start *Node) <-chan *Node {
	ch := make(chan *Node)
	walked := NewNodeSet()

	go func() {
		walkGraph(start, ch, walked)
		close(ch)
	}()
	return ch
}

func walkGraph(start *Node, ch chan *Node, walked NodeSet) {
	defer func() {
		walked.Add(start)
	}()
	if _, ok := walked[start]; ok {
		return // walked before
	}

	ch <- start

	for _, child := range start.children {
		walkGraph(child, ch, walked)
	}

}

// Sort topologically sorts a ExprGraph: root of graph will be first
// nodes are sorted using gonum's SortStabilized function.
//
// see https://godoc.org/gonum.org/v1/gonum/graph/topo#SortStabilized for more info
func Sort(g *ExprGraph) (sorted Nodes, err error) {
	var sortedNodes []graph.Node
	// if sortedNodes, err = topo.Sort(g); err != nil {
	if sortedNodes, err = topo.SortStabilized(g, reverseLexical); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	sorted = graphNodeToNode(iterator.NewOrderedNodes(sortedNodes))
	return
}

// UnstableSort performs a topological sort of the directed graph g returning the 'from' to 'to'
// sort order. If a topological ordering is not possible, an Unorderable error is returned
// listing cyclic components in g with each cyclic component's members sorted by ID. When
// an Unorderable error is returned, each cyclic component's topological position within
// the sorted nodes is marked with a nil graph.Node.
func UnstableSort(g *ExprGraph) (sorted Nodes, err error) {
	var sortedNodes []graph.Node
	if sortedNodes, err = topo.Sort(g); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	sorted = graphNodeToNode(iterator.NewOrderedNodes(sortedNodes))
	return
}

func reverseNodes(sorted Nodes) {
	for i := len(sorted)/2 - 1; i >= 0; i-- {
		j := len(sorted) - i - 1
		sorted[i], sorted[j] = sorted[j], sorted[i]
	}
}

type byID []graph.Node

func (ns byID) Len() int           { return len(ns) }
func (ns byID) Less(i, j int) bool { return ns[i].ID() > ns[j].ID() }
func (ns byID) Swap(i, j int)      { ns[i], ns[j] = ns[j], ns[i] }

func reverseLexical(a []graph.Node) {
	sort.Sort(byID(a))
}
