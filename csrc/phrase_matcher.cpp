// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr const char* capsule_name = "sonar.phrase_automaton";

struct Node {
  std::unordered_map<unsigned char, int> edges;
  int failure = 0;
  int depth = 0;
  int terminal = 0;
};

struct Automaton {
  std::vector<Node> nodes{1};

  int advance(int state, unsigned char byte) const {
    while (true) {
      const auto it = nodes[state].edges.find(byte);
      if (it != nodes[state].edges.end()) return it->second;
      if (state == 0) return 0;
      state = nodes[state].failure;
    }
  }

  void add(const std::string& pattern) {
    int state = 0;
    for (unsigned char byte : pattern) {
      auto result = nodes[state].edges.emplace(byte, nodes.size());
      const int next = result.first->second;
      if (result.second) {
        const int depth = nodes[state].depth + 1;
        nodes.emplace_back();
        nodes.back().depth = depth;
      }
      state = next;
    }
    nodes[state].terminal = static_cast<int>(pattern.size());
  }

  void finish() {
    std::queue<int> queue;
    for (const auto& edge : nodes[0].edges) queue.push(edge.second);
    while (!queue.empty()) {
      const int parent = queue.front();
      queue.pop();
      for (const auto& [byte, child] : nodes[parent].edges) {
        const int failure = advance(nodes[parent].failure, byte);
        nodes[child].failure = failure;
        nodes[child].terminal =
            std::max(nodes[child].terminal, nodes[failure].terminal);
        queue.push(child);
      }
    }
  }
};

void destroy(PyObject* capsule) {
  delete static_cast<Automaton*>(PyCapsule_GetPointer(capsule, capsule_name));
}

PyObject* compile(PyObject*, PyObject* patterns) {
  const Py_ssize_t count = PySequence_Size(patterns);
  if (count < 0) return nullptr;
  if (count == 0 || count > 2048) {
    PyErr_SetString(PyExc_ValueError, "Expected 1..2048 byte patterns");
    return nullptr;
  }
  try {
    auto automaton = std::make_unique<Automaton>();
    Py_ssize_t total_bytes = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
      PyObject* item = PySequence_GetItem(patterns, i);
      if (!item) return nullptr;
      char* data;
      Py_ssize_t length;
      if (PyBytes_AsStringAndSize(item, &data, &length) < 0) {
        Py_DECREF(item);
        return nullptr;
      }
      total_bytes += length;
      if (length == 0 || total_bytes > 262144) {
        Py_DECREF(item);
        PyErr_SetString(PyExc_ValueError,
                        "Empty pattern or phrase byte budget exceeded");
        return nullptr;
      }
      // Release the Python reference even if trie allocation throws.
      std::string pattern;
      try {
        pattern.assign(data, length);
      } catch (...) {
        Py_DECREF(item);
        throw;
      }
      Py_DECREF(item);
      automaton->add(pattern);
    }
    automaton->finish();
    PyObject* capsule = PyCapsule_New(automaton.get(), capsule_name, destroy);
    if (capsule) automaton.release();
    return capsule;
  } catch (const std::bad_alloc&) {
    return PyErr_NoMemory();
  }
}

PyObject* scan(PyObject*, PyObject* args) {
  PyObject* capsule;
  int state;
  const char* data;
  Py_ssize_t length;
  if (!PyArg_ParseTuple(args, "Oiy#", &capsule, &state, &data, &length))
    return nullptr;
  const auto* automaton = static_cast<const Automaton*>(
      PyCapsule_GetPointer(capsule, capsule_name));
  if (!automaton) return nullptr;
  if (state < 0 || static_cast<size_t>(state) >= automaton->nodes.size()) {
    PyErr_SetString(PyExc_ValueError, "Invalid phrase automaton state");
    return nullptr;
  }
  bool found = false;
  Py_ssize_t earliest = 0;
  for (Py_ssize_t i = 0; i < length; ++i) {
    state = automaton->advance(state, static_cast<unsigned char>(data[i]));
    const int terminal = automaton->nodes[state].terminal;
    if (terminal) {
      const Py_ssize_t start = i + 1 - terminal;
      if (!found || start < earliest) earliest = start;
      found = true;
    }
  }
  PyObject* match;
  if (found) {
    match = PyLong_FromSsize_t(earliest);
    if (!match) return nullptr;
  } else {
    match = Py_None;
    Py_INCREF(match);
  }
  return Py_BuildValue("(iNn)", state, match,
                       static_cast<Py_ssize_t>(automaton->nodes[state].depth));
}

PyMethodDef methods[] = {
    {"compile", compile, METH_O, nullptr},
    {"scan", scan, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT,
                      "_phrase_matcher",
                      nullptr,
                      -1,
                      methods,
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__phrase_matcher() { return PyModule_Create(&module); }
