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

  final List<Uint8List> _images = [];
  bool _modelLoading = true;
  String? _modelError;
  bool _comparing = false;
  SimilarImagesResult? _similarResult;
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

  Future<void> _addImage() async {
    final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null || !mounted) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _images.add(bytes);
      _similarResult = null;
      _compareError = null;
    });
  }

  void _removeImage(int index) {
    setState(() {
      _images.removeAt(index);
      _similarResult = null;
      _compareError = null;
    });
  }

  Future<void> _findSimilar() async {
    if (_images.length < 2) {
      setState(() => _compareError = 'Add at least 2 images to find similar ones.');
      return;
    }
    if (!_similarityService.isLoaded) {
      setState(() => _compareError = 'Model not loaded');
      return;
    }
    setState(() {
      _comparing = true;
      _compareError = null;
      _similarResult = null;
    });
    await Future.delayed(Duration.zero);
    if (!mounted) return;
    try {
      final result = await Future(
        () => _similarityService.findSimilarImages(_images, threshold: 0.95),
      );
      if (mounted) {
        setState(() {
          _comparing = false;
          _similarResult = result;
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
                            'Add images, then tap "Find similar" to see which ones are similar.',
                            style: TextStyle(fontSize: 14),
                          ),
                          const SizedBox(height: 16),
                          LayoutBuilder(
                            builder: (context, constraints) {
                              const crossAxisCount = 2;
                              const spacing = 12.0;
                              final size = (constraints.maxWidth - spacing * (crossAxisCount - 1)) / crossAxisCount;
                              return Wrap(
                                spacing: spacing,
                                runSpacing: spacing,
                                children: [
                                  ...List.generate(_images.length, (i) {
                                    return SizedBox(
                                      width: size,
                                      height: size,
                                      child: _ImageSlot(
                                        bytes: _images[i],
                                        label: '${i + 1}',
                                        onTap: () {},
                                        onRemove: () => _removeImage(i),
                                        showRemove: true,
                                      ),
                                    );
                                  }),
                                  SizedBox(
                                    width: size,
                                    height: size,
                                    child: _ImageSlot(
                                      bytes: null,
                                      label: '+',
                                      onTap: _addImage,
                                      showRemove: false,
                                    ),
                                  ),
                                ],
                              );
                            },
                          ),
                          const SizedBox(height: 24),
                          FilledButton.icon(
                            onPressed: _comparing ? null : _findSimilar,
                            icon: _comparing
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(strokeWidth: 2),
                                  )
                                : const Icon(Icons.compare),
                            label: Text(_comparing ? 'Comparing...' : 'Find similar'),
                          ),
                          if (_compareError != null) ...[
                            const SizedBox(height: 12),
                            Text(
                              _compareError!,
                              style: TextStyle(color: Theme.of(context).colorScheme.error),
                            ),
                          ],
                          if (_similarResult != null) ...[
                            const SizedBox(height: 24),
                            _SimilarResultView(result: _similarResult!, imageCount: _images.length),
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

class _SimilarResultView extends StatelessWidget {
  const _SimilarResultView({required this.result, required this.imageCount});

  final SimilarImagesResult result;
  final int imageCount;

  @override
  Widget build(BuildContext context) {
    if (result.pairs.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Text(
            'No similar images found among the $imageCount image(s) (threshold 95%). Try adding more images.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
      );
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Similar images',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            ...result.pairs.map((p) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Row(
                    children: [
                      Icon(Icons.check_circle, size: 20, color: Theme.of(context).colorScheme.primary),
                      const SizedBox(width: 8),
                      Text(
                        'Image ${p.indexA + 1} & Image ${p.indexB + 1}: ${(p.score * 100).toStringAsFixed(0)}% similar',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                )),
            if (result.groups.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(
                'Groups (similar to each other)',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              const SizedBox(height: 8),
              ...result.groups.asMap().entries.map((e) {
                final group = e.value;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(
                    'Group ${e.key + 1}: ${group.map((i) => 'Image ${i + 1}').join(', ')}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                );
              }),
            ],
          ],
        ),
      ),
    );
  }
}

class _ImageSlot extends StatelessWidget {
  const _ImageSlot({
    required this.bytes,
    required this.label,
    required this.onTap,
    required this.showRemove,
    this.onRemove,
  });

  final Uint8List? bytes;
  final String label;
  final VoidCallback onTap;
  final bool showRemove;
  final VoidCallback? onRemove;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            height: double.infinity,
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
                        Icon(
                          Icons.add_photo_alternate,
                          size: 40,
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                        const SizedBox(height: 4),
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
                            child: Text(
                              label,
                              style: const TextStyle(color: Colors.white, fontSize: 12),
                            ),
                          ),
                        ),
                      ],
                    ),
            ),
          ),
          if (showRemove && bytes != null && onRemove != null)
            Positioned(
              top: -6,
              right: -6,
              child: Material(
                color: Theme.of(context).colorScheme.errorContainer,
                shape: const CircleBorder(),
                child: InkWell(
                  onTap: onRemove,
                  customBorder: const CircleBorder(),
                  child: const Padding(
                    padding: EdgeInsets.all(6),
                    child: Icon(Icons.close, size: 18),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
