import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kidney Stone Detector',
      theme: ThemeData(primarySwatch: Colors.teal),
      home: const DetectionPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  File? _image;
  String _result = '';
  bool _loading = false;
  int _selectedIndex = 0;

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() {
        _image = File(picked.path);
        _result = '';
      });
    }
  }

  Future<void> _uploadAndPredict() async {
    if (_image == null) return;
    setState(() => _loading = true);

    final uri = Uri.parse(
      'http://10.0.2.2:5000/predict',
    ); // for Android emulator
    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      await http.MultipartFile.fromPath(
        'image',
        _image!.path,
        contentType: MediaType('image', 'jpeg'),
      ),
    );

    final response = await request.send();
    final respStr = await response.stream.bytesToString();
    final jsonResp = json.decode(respStr);

    setState(() {
      _loading = false;
      _result = jsonResp['result'] ?? 'No result';
    });
  }

  void _onNavTapped(int index) {
    setState(() {
      _selectedIndex = index;
      // Optional: Add page navigation logic here
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Center(child: Text("Kidney Stone Detector"))),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_image != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(_image!, height: 200),
                ),
              const SizedBox(height: 30),
              ElevatedButton(
                onPressed: _pickImage,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(200, 48),
                ),
                child: const Text("Pick Scan Image"),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _uploadAndPredict,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(200, 48),
                ),
                child:
                    _loading
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Text("Detect"),
              ),
              const SizedBox(height: 30),
              Text(
                _result,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onNavTapped,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ],
        selectedItemColor: Colors.teal,
      ),
    );
  }
}
