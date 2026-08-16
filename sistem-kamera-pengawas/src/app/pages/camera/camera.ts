import { Component, DestroyRef, inject, OnInit } from '@angular/core';
import { DecimalPipe, NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { startWith, switchMap } from 'rxjs/operators';

const API = 'http://localhost:8002';
const AI  = 'http://localhost:8001';

interface Camera {
  id:            number;
  camera_id:     string | null;
  name:          string;
  zone_location: string | null;
  floor:         string | null;
  rtsp_url:      string;
  is_active:     boolean;
}

@Component({
  selector: 'app-camera',
  standalone: true,
  imports: [NgClass, DecimalPipe],
  templateUrl: './camera.html',
  styleUrl: './camera.css',
})
export class CameraPage implements OnInit {
  cameras: Camera[] = [];

  // Camera modal
  showModal       = false;
  editingCamera: Camera | null = null;
  formName        = '';
  formFloor       = '';
  formRtspUrl     = '';
  formIsActive    = true;

  // RTSP show/hide per row
  visibleRtsp = new Set<number>();

  // Camera table filters
  floorFilter:  string = 'semua';
  statusFilter: 'semua' | 'aktif' | 'nonaktif' = 'semua';

  // Stream state
  streamRunning   = false;
  streamLoading   = false;
  framesProcessed = 0;

  private destroyRef = inject(DestroyRef);

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadCameras();
    this._pollStreamStatus();
  }

  private _pollStreamStatus(): void {
    interval(3000).pipe(
      startWith(0),
      switchMap(() => this.http.get<{ running: boolean; frames_processed: number }>(
        `${AI}/stream/status`
      )),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: s => {
        this.streamRunning   = s.running;
        this.framesProcessed = s.frames_processed;
      },
      error: () => { this.streamRunning = false; },
    });
  }

  startStream(): void {
    if (this.streamLoading) return;
    this.streamLoading = true;
    this.http.post(`${AI}/stream/start`, {}).subscribe({
      next: () => { this.streamRunning = true;  this.streamLoading = false; },
      error: ()=> { this.streamLoading = false; },
    });
  }

  stopStream(): void {
    if (this.streamLoading) return;
    this.streamLoading = true;
    this.http.post(`${AI}/stream/stop`, {}).subscribe({
      next: () => { this.streamRunning = false; this.streamLoading = false; },
      error: ()=> { this.streamLoading = false; },
    });
  }

  loadCameras(): void {
    this.http.get<Camera[]>(`${API}/cameras`).subscribe({
      next:  cams => { this.cameras = cams; },
      error: err => console.error('[camera] load cameras error:', err),
    });
  }

  get floors(): string[] {
    const set = new Set(this.cameras.map(c => c.floor).filter((f): f is string => !!f));
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }

  get filteredCameras(): Camera[] {
    return this.cameras.filter(c => {
      const matchFloor  = this.floorFilter === 'semua' || c.floor === this.floorFilter;
      const matchStatus =
        this.statusFilter === 'semua' ||
        (this.statusFilter === 'aktif'    && c.is_active) ||
        (this.statusFilter === 'nonaktif' && !c.is_active);
      return matchFloor && matchStatus;
    });
  }

  onFloorFilterChange(event: Event): void {
    this.floorFilter = (event.target as HTMLSelectElement).value;
  }

  onStatusFilterChange(event: Event): void {
    this.statusFilter = (event.target as HTMLSelectElement).value as 'semua' | 'aktif' | 'nonaktif';
  }

  toggleCamera(cam: Camera, event: Event): void {
    event.stopPropagation();
    this.http.patch<Camera>(`${API}/cameras/${cam.id}/toggle`, {}).subscribe(updated => {
      cam.is_active = updated.is_active;
    });
  }

  deleteCamera(cam: Camera, event: Event): void {
    event.stopPropagation();
    if (!confirm(`Hapus kamera "${cam.name}"?`)) return;
    this.http.delete(`${API}/cameras/${cam.id}`).subscribe(() => {
      this.cameras = this.cameras.filter(c => c.id !== cam.id);
    });
  }

  openAddModal(): void {
    this.editingCamera = null;
    this.formName      = '';
    this.formFloor     = '';
    this.formRtspUrl   = '';
    this.formIsActive  = true;
    this.showModal     = true;
  }

  openEditModal(cam: Camera, event: Event): void {
    event.stopPropagation();
    this.editingCamera = cam;
    this.formName      = cam.name;
    this.formFloor     = cam.floor ?? '';
    this.formRtspUrl   = cam.rtsp_url;
    this.formIsActive  = cam.is_active;
    this.showModal     = true;
  }

  saveCamera(): void {
    const body: Record<string, unknown> = {
      name:     this.formName.trim(),
      floor:    this.formFloor.trim() || null,
      rtsp_url: this.formRtspUrl.trim(),
    };
    if (!this.editingCamera) {
      body['is_active'] = true;
    } else {
      body['is_active'] = this.editingCamera.is_active;
    }
    const req = this.editingCamera
      ? this.http.put<Camera>(`${API}/cameras/${this.editingCamera.id}`, body)
      : this.http.post<Camera>(`${API}/cameras`, body);

    req.subscribe({
      next: () => {
        this.showModal = false;
        this.loadCameras();
      },
      error: (err) => {
        console.error('[camera] saveCamera error:', err);
        alert('Gagal menyimpan kamera. Periksa koneksi ke backend.');
      },
    });
  }

  toggleRtspVisibility(id: number, event: Event): void {
    event.stopPropagation();
    if (this.visibleRtsp.has(id)) {
      this.visibleRtsp.delete(id);
    } else {
      this.visibleRtsp.add(id);
    }
  }

  maskedUrl(url: string): string {
    try {
      const u = new URL(url);
      if (u.password) u.password = '••••••';
      return u.toString();
    } catch {
      return url.replace(/:([^@]+)@/, ':••••••@');
    }
  }

  closeModal(): void { this.showModal = false; }

  formInput(field: 'name' | 'floor' | 'rtspUrl', event: Event): void {
    const val = (event.target as HTMLInputElement).value;
    if (field === 'name')    this.formName    = val;
    if (field === 'floor')   this.formFloor   = val;
    if (field === 'rtspUrl') this.formRtspUrl = val;
  }
}
