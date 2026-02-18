import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'services/image_similarity_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Similarity',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ImageSimilarityPage(),
    );
  }
}

class ImageSimilarityPage extends StatefulWidget {
  const ImageSimilarityPage({super.key});

  @override
  State<ImageSimilarityPage> createState() => _ImageSimilarityPageState();
}

class _ImageSimilarityPageState extends State<ImageSimilarityPage> {
  final ImageSimilarityService _similarityService = ImageSimilarityService();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _image1;
  Uint8List? _image2;
  bool _modelLoading = true;
  String? _modelError;
  bool _comparing = false;
  double? _similarityScore;
  String? _compareError;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _similarityService.loadModel();
      if (mounted) {
        setState(() {
          _modelLoading = false;
          _modelError = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _modelLoading = false;
          _modelError = 'Failed to load model: $e';
        });
      }
    }
  }

  @override
  void dispose() {
    _similarityService.close();
    super.dispose();
  }

  Future<void> _pickImage(bool isFirst) async {
    final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null || !mounted) return;
    final bytes = await file.readAsBytes();
    setState(() {
      if (isFirst) {
        _image1 = bytes;
      } else {
        _image2 = bytes;
      }
      _similarityScore = null;
      _compareError = null;
    });
  }

  Future<void> _compare() async {
    if (_image1 == null || _image2 == null) {
      setState(() {
        _compareError = 'Pick both images first';
      });
      return;
    }
    if (!_similarityService.isLoaded) {
      setState(() => _compareError = 'Model not loaded');
      return;
    }
    setState(() {
      _comparing = true;
      _compareError = null;
      _similarityScore = null;
    });
    // Yield so the loading UI can paint before we run heavy inference.
    await Future.delayed(Duration.zero);
    if (!mounted) return;
    try {
      final score = await Future(() => _similarityService.getSimilarity(_image1!, _image2!));
      if (mounted) {
        setState(() {
          _comparing = false;
          _similarityScore = score;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _comparing = false;
          _compareError = '$e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Similarity'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _modelLoading
          ? const Center(child: CircularProgressIndicator())
          : _modelError != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          _modelError!,
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Theme.of(context).colorScheme.error),
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Model is loaded via tflite_flutter from assets/models/mobilenet_quant.tflite (same as package example).',
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                )
              : Stack(
                  children: [
                    SingleChildScrollView(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const Text(
                            'Pick two images to compare using a TFLite embedding model.',
                            style: TextStyle(fontSize: 14),
                          ),
                          const SizedBox(height: 24),
                          Row(
                        children: [
                          Expanded(
                            child: _ImageSlot(
                              bytes: _image1,
                              label: 'Image 1',
                              onTap: () => _pickImage(true),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: _ImageSlot(
                              bytes: _image2,
                              label: 'Image 2',
                              onTap: () => _pickImage(false),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 24),
                      FilledButton.icon(
                        onPressed: _comparing ? null : _compare,
                        icon: _comparing
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Icon(Icons.compare),
                        label: Text(_comparing ? 'Comparing...' : 'Compare'),
                      ),
                      if (_compareError != null) ...[
                        const SizedBox(height: 12),
                        Text(
                          _compareError!,
                          style: TextStyle(color: Theme.of(context).colorScheme.error),
                        ),
                      ],
                      if (_similarityScore != null) ...[
                        const SizedBox(height: 24),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(20),
                            child: Column(
                              children: [
                                Text(
                                  'Similarity',
                                  style: Theme.of(context).textTheme.titleMedium,
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  '${(_similarityScore! * 100).toStringAsFixed(1)}%',
                                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                                        color: Theme.of(context).colorScheme.primary,
                                      ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _similarityScore! >= 0.7
                                      ? 'Similar'
                                      : _similarityScore! >= 0.4
                                          ? 'Somewhat similar'
                                          : 'Different',
                                  style: Theme.of(context).textTheme.bodyMedium,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                    if (_comparing)
                      Positioned.fill(
                        child: Container(
                          color: Colors.black26,
                          child: const Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                CircularProgressIndicator(),
                                SizedBox(height: 16),
                                Text('Comparing...', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
    );
  }
}

class _ImageSlot extends StatelessWidget {
  const _ImageSlot({
    required this.bytes,
    required this.label,
    required this.onTap,
  });

  final Uint8List? bytes;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        height: 160,
        decoration: BoxDecoration(
          border: Border.all(color: Theme.of(context).colorScheme.outline),
          borderRadius: BorderRadius.circular(12),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: bytes == null
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add_photo_alternate, size: 48, color: Theme.of(context).colorScheme.onSurfaceVariant),
                    const SizedBox(height: 8),
                    Text(label, style: Theme.of(context).textTheme.bodyMedium),
                  ],
                )
              : Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(bytes!, fit: BoxFit.cover),
                    Positioned(
                      left: 0,
                      right: 0,
                      bottom: 0,
                      child: Container(
                        padding: const EdgeInsets.all(6),
                        color: Colors.black54,
                        child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}
