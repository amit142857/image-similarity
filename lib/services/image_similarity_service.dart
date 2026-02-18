import 'dart:math' show sqrt;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// A pair of image indices and their similarity score (0–1).
class SimilarPair {
  const SimilarPair({required this.indexA, required this.indexB, required this.score});
  final int indexA;
  final int indexB;
  final double score;
}

/// Result of finding similar images: pairs above threshold and groups (connected components).
class SimilarImagesResult {
  const SimilarImagesResult({required this.pairs, required this.groups});
  final List<SimilarPair> pairs;
  /// Each group is a list of image indices that are similar to each other.
  final List<List<int>> groups;
}

/// Service that uses a TFLite model (from tflite_flutter package) to compare images
/// and compute similarity (cosine similarity of feature/classification vectors).
class ImageSimilarityService {
  Interpreter? _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;
  TensorType? _inputType;

  /// Model path: try package asset first (when using a tflite_flutter build that
  /// includes it), then app asset. Same model as tflite_flutter example.
  static const String _modelPathPackage =
      'packages/tflite_flutter/assets/models/mobilenet_quant.tflite';
  static const String _modelPathApp = 'assets/models/mobilenet_quant.tflite';

  bool get isLoaded => _interpreter != null;

  /// Loads the TFLite model using tflite_flutter (from package or app asset).
  /// Call this once before [getSimilarity] (e.g. at app startup).
  Future<void> loadModel() async {
    if (_interpreter != null) return;
    try {
      _interpreter = await Interpreter.fromAsset(_modelPathPackage);
    } catch (_) {
      _interpreter = await Interpreter.fromAsset(_modelPathApp);
    }
    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);
    _inputShape = List<int>.from(inputTensor.shape);
    _outputShape = List<int>.from(outputTensor.shape);
    _inputType = inputTensor.type;
  }

  /// Releases the interpreter. Call when done (e.g. on dispose).
  void close() {
    _interpreter?.close();
    _interpreter = null;
    _inputShape = null;
    _outputShape = null;
    _inputType = null;
  }

  /// Preprocesses image bytes to a float tensor matching the model input shape.
  /// Expects shape like [1, height, width, 3] with values in [0, 1].
  Float32List _preprocessImage(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      throw Exception('Failed to decode image');
    }
    // Input shape is typically [1, H, W, 3]
    final batch = _inputShape![0];
    final height = _inputShape![1];
    final width = _inputShape![2];
    final channels = _inputShape!.length > 3 ? _inputShape![3] : 3;

    final resized = img.copyResize(
      decoded,
      width: width,
      height: height,
      interpolation: img.Interpolation.linear,
    );

    final total = batch * height * width * channels;
    final input = Float32List(total);
    int idx = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = resized.getPixel(x, y);
        // Normalize to [0, 1]. Model may expect [0,255] for uint8; we use float [0,1].
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }
    return input;
  }

  /// Builds an output buffer matching the interpreter's output tensor shape (e.g. [1, 1001]).
  static List<dynamic> _outputBufferFromShape(List<int> shape) {
    if (shape.length == 1) {
      return List<double>.filled(shape[0], 0.0);
    }
    return List.generate(
      shape[0],
      (_) => _outputBufferFromShape(shape.sublist(1)),
    );
  }

  /// Runs the model on one image and returns the embedding/classification vector.
  List<double> _getEmbedding(Uint8List imageBytes) {
    if (_interpreter == null || _inputShape == null || _outputShape == null) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }
    final input = _preprocessImage(imageBytes);
    final output = _outputBufferFromShape(_outputShape!);

    if (_inputType == TensorType.uint8) {
      final quantized = Uint8List.fromList(
        input.map((e) => (e.clamp(0.0, 1.0) * 255).round()).toList(),
      );
      _interpreter!.run(quantized, output);
    } else {
      _interpreter!.run(input, output);
    }
    // Flatten to 1D List<double> (e.g. [1, 1001] -> take output[0]; output is List<dynamic>)
    return _flattenToDoubleList(output);
  }

  /// Converts interpreter output (nested List<dynamic>) to List<double>.
  static List<double> _flattenToDoubleList(dynamic raw) {
    if (raw is List) {
      if (raw.isEmpty) return [];
      final first = raw.first;
      if (first is List) {
        return _flattenToDoubleList(first);
      }
      return raw.map((e) => (e as num).toDouble()).toList();
    }
    return [(raw as num).toDouble()];
  }

  /// Computes cosine similarity between two vectors (range -1 to 1; 1 = identical).
  static double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vectors must have same length');
    }
    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    final denom = sqrt(normA * normB);
    if (denom == 0) return 0;
    return dot / denom;
  }

  /// Compares two images and returns a similarity score in [0, 1].
  /// 1.0 means very similar, 0.0 means very different.
  double getSimilarity(Uint8List image1Bytes, Uint8List image2Bytes) {
    final emb1 = _getEmbedding(image1Bytes);
    final emb2 = _getEmbedding(image2Bytes);
    final cos = _cosineSimilarity(emb1, emb2);
    return (cos + 1) / 2;
  }

  /// Finds similar images among [images]. Returns pairs with score >= [threshold]
  /// and groups (connected components of similar images).
  /// [threshold] is in [0, 1]; default 0.95 (95%).
  SimilarImagesResult findSimilarImages(
    List<Uint8List> images, {
    double threshold = 0.95,
  }) {
    if (images.length < 2) {
      return const SimilarImagesResult(pairs: [], groups: []);
    }
    final embeddings = <int, List<double>>{};
    for (var i = 0; i < images.length; i++) {
      embeddings[i] = _getEmbedding(images[i]);
    }
    final pairs = <SimilarPair>[];
    for (var i = 0; i < images.length; i++) {
      for (var j = i + 1; j < images.length; j++) {
        final cos = _cosineSimilarity(embeddings[i]!, embeddings[j]!);
        final score = (cos + 1) / 2;
        if (score >= threshold) {
          pairs.add(SimilarPair(indexA: i, indexB: j, score: score));
        }
      }
    }
    final groups = _buildGroups(images.length, pairs);
    return SimilarImagesResult(pairs: pairs, groups: groups);
  }

  /// Builds groups (connected components) from similar pairs using union-find.
  static List<List<int>> _buildGroups(int n, List<SimilarPair> pairs) {
    final parent = List<int>.generate(n, (i) => i);
    int find(int x) {
      if (parent[x] != x) parent[x] = find(parent[x]);
      return parent[x];
    }
    void union(int a, int b) {
      parent[find(a)] = find(b);
    }
    for (final p in pairs) {
      union(p.indexA, p.indexB);
    }
    final map = <int, List<int>>{};
    for (var i = 0; i < n; i++) {
      map.putIfAbsent(find(i), () => []).add(i);
    }
    return map.values.where((g) => g.length > 1).toList();
  }
}
