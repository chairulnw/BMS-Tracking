import { Component, DestroyRef, inject, OnInit } from '@angular/core';
import { NgClass, SlicePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { startWith, switchMap } from 'rxjs/operators';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';

const API    = 'http://localhost:8002';
const AI_API = 'http://localhost:8001';

interface BackendPerson {
  id:                  number;
  name:                string;
  label:               string;
  first_seen:          string | null;
  last_seen:           string | null;
  last_camera:         string | null;
  best_thumbnail_url:  string | null;
  is_known:            boolean;
  enrollment_date:     string | null;
}

interface OccupancyRoom {
  camera_id:         string;
  room_name:         string;
  floor:             string | null;
  count_in:          number;
  count_out:         number;
  current_occupancy: number;
}

interface AmbiguousCase {
  id:           string;
  track_id:     number;
  cam_id:       string;
  candidate:    string;
  sim_top1:     number;
  sim_top2:     number;
  margin:       number;
  threshold:    number;
  timestamp:    string;
  snapshot_url: string | null;
}

interface Movement {
  timestamp:    string;
  time:         string;
  location:     string;
  floor:        string;
  note:         string;
  camera:       string;
  isCurrent:    boolean;
  event_kind:   string | null;
  direction:    string | null;
  snapshot_url: string | null;
  track_id:     number | null;
}

interface Person {
  id:             number;
  name:           string;
  label:          string;
  jabatan:        string;
  status:         'terdaftar' | 'unknown';
  lastCamera:     string;
  thumbnailUrl:   string | null;
  time:           string;
  firstSeen:      string;
  lastSeen:       string;
  lastSeenRaw:    string | null;
  enrollmentDate: string | null;
  movements:      Movement[];
}

type Filter = 'semua' | 'terdaftar' | 'unknown';
type SortMode = 'terbaru' | 'nama';

@Component({
  selector: 'app-people',
  standalone: true,
  imports: [NgClass, SlicePipe, AuthUrlPipe],
  templateUrl: './people.html',
  styleUrl: './people.css',
})
export class People implements OnInit {
  private http       = inject(HttpClient);
  private destroyRef = inject(DestroyRef);

  activeFilter: Filter = 'semua';
  sortMode: SortMode   = 'terbaru';
  searchQuery  = '';

  feedOnline  = false;
  editMode    = false;
  editName    = '';
  editJabatan = '';

  people: Person[]              = [];
  selectedPerson: Person | null = null;
  occupancy: OccupancyRoom[]    = [];
  ambiguous: AmbiguousCase[]    = [];
  assigningAmbId: string | null = null;

  // URL gambar besar di panel kanan — diperbarui via tombol "Lihat"
  previewUrl: string | null = null;
  // Index movement yang sedang di-preview (null = belum ada pilihan manual)
  previewMvIdx: number | null = null;

  renamingId  = -1;
  renameInput = '';

  selectedDate: string = this._todayISO();

  ngOnInit(): void {
    interval(5000).pipe(
      startWith(0),
      switchMap(() => this.http.get<BackendPerson[]>(`${API}/persons?date=${this.selectedDate}`)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next:  records => this.syncPeople(records),
      error: err     => console.error('[people] polling error:', err),
    });

    interval(5000).pipe(
      startWith(0),
      switchMap(() => this.http.get<OccupancyRoom[]>(`${API}/occupancy?date=${this.selectedDate}`)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next:  data => this.occupancy = data,
      error: err  => console.error('[people] occupancy error:', err),
    });

    // interval(5000).pipe(
    //   startWith(0),
    //   switchMap(() => this.http.get<AmbiguousCase[]>(`${AI_API}/stream/ambiguous`)),
    //   takeUntilDestroyed(this.destroyRef),
    // ).subscribe({
    //   next:  data => this.ambiguous = data,
    //   error: ()   => { /* AI service mungkin belum jalan */ },
    // });
  }

  onDateChange(event: Event): void {
    const val = (event.target as HTMLInputElement).value;
    if (!val) return;
    this.selectedDate  = val;
    this.selectedPerson = null;
    this.people        = [];
    this.http.get<BackendPerson[]>(`${API}/persons?date=${this.selectedDate}`).subscribe({
      next:  records => this.syncPeople(records),
      error: err     => console.error('[people] date fetch error:', err),
    });
    this.http.get<OccupancyRoom[]>(`${API}/occupancy?date=${this.selectedDate}`).subscribe({
      next:  data => this.occupancy = data,
      error: err  => console.error('[people] occupancy date fetch error:', err),
    });
  }

  private syncPeople(records: BackendPerson[]): void {
    const prevId = this.selectedPerson?.id;

    this.people = records.map(r => {
      const existing = this.people.find(p => p.id === r.id);
      return {
        id:             r.id,
        name:           r.name,
        label:          r.label,
        jabatan:        existing?.jabatan ?? '',
        status:         r.is_known ? 'terdaftar' : 'unknown',
        lastCamera:     r.last_camera ?? '—',
        thumbnailUrl:   r.best_thumbnail_url ?? null,
        time:           this._fmt(r.last_seen),
        firstSeen:      this._fmtFull(r.first_seen),
        lastSeen:       this._fmtFull(r.last_seen),
        lastSeenRaw:    r.last_seen,
        enrollmentDate: r.enrollment_date ?? null,
        movements:      existing?.movements ?? [],
      };
    });

    if (prevId != null) {
      this.selectedPerson =
        this.people.find(p => p.id === prevId) ?? this.people[0] ?? null;
    } else {
      this.selectedPerson = this.people[0] ?? null;
      this.previewUrl     = this.selectedPerson?.thumbnailUrl ?? null;
    }

    if (this.selectedPerson) {
      this._fetchMovements(this.selectedPerson);
    }
  }

  // ── Inline rename ────────────────────────────────────────────────────────────

  startRename(person: Person, event: Event): void {
    event.stopPropagation();
    this.renamingId  = person.id;
    this.renameInput = person.name;
  }

  confirmRename(person: Person, event: Event): void {
    event.stopPropagation();
    const newName = this.renameInput.trim();
    if (!newName || newName === person.name) { this.cancelRename(); return; }

    this.http.patch<BackendPerson>(
      `${API}/persons/${person.id}/rename`,
      { new_name: newName },
    ).subscribe({
      next: updated => {
        person.name   = updated.name;
        person.status = updated.is_known ? 'terdaftar' : 'unknown';
        if (this.selectedPerson?.id === person.id) {
          this.selectedPerson.name   = updated.name;
          this.selectedPerson.status = person.status;
        }
        this.cancelRename();
      },
      error: err => { console.error('[people] rename error:', err); this.cancelRename(); },
    });
  }

  // ── Ambiguous ReID resolution ────────────────────────────────────────────────

  resolveAmbiguous(amb: AmbiguousCase, action: 'confirm' | 'reject'): void {
    this.http.post<{ status: string }>(
      `${AI_API}/stream/ambiguous/${amb.id}/resolve?action=${action}`, {}
    ).subscribe({
      next:  () => this._removeAmb(amb.id),
      error: err => console.error('[people] resolve error:', err),
    });
  }

  assignAmbiguous(amb: AmbiguousCase, person: Person): void {
    this.http.post<{ status: string }>(
      `${AI_API}/stream/ambiguous/${amb.id}/resolve?action=assign&target=${encodeURIComponent(person.label)}`, {}
    ).subscribe({
      next:  () => this._removeAmb(amb.id),
      error: err => console.error('[people] assign error:', err),
    });
  }

  toggleAssignPicker(ambId: string): void {
    this.assigningAmbId = this.assigningAmbId === ambId ? null : ambId;
  }

  private _removeAmb(ambId: string): void {
    this.ambiguous     = this.ambiguous.filter(a => a.id !== ambId);
    this.assigningAmbId = null;
  }

  ambSnapshotUrl(amb: AmbiguousCase): string {
    return `${AI_API}/stream/ambiguous/${amb.id}/snapshot`;
  }

  cancelRename(): void {
    this.renamingId  = -1;
    this.renameInput = '';
  }

  // ── Camera overlay edit ──────────────────────────────────────────────────────

  openEdit(): void {
    if (!this.selectedPerson) return;
    this.editName    = this.selectedPerson.name;
    this.editJabatan = this.selectedPerson.jabatan;
    this.editMode    = true;
  }

  saveEdit(): void {
    if (!this.selectedPerson) return;
    const newName = this.editName.trim() || this.selectedPerson.name;

    if (newName !== this.selectedPerson.name) {
      this.http.patch<BackendPerson>(
        `${API}/persons/${this.selectedPerson.id}/rename`,
        { new_name: newName },
      ).subscribe({ error: err => console.error('[people] rename error:', err) });
    }

    this.selectedPerson.name    = newName;
    this.selectedPerson.jabatan = this.editJabatan.trim();
    if (this.selectedPerson.jabatan) this.selectedPerson.status = 'terdaftar';
    this.editMode = false;
  }

  cancelEdit(): void { this.editMode = false; }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  get filteredPeople(): Person[] {
    const filtered = this.people.filter(p => {
      const matchFilter = this.activeFilter === 'semua' || p.status === this.activeFilter;
      const matchSearch = p.name.toLowerCase().includes(this.searchQuery.toLowerCase());
      return matchFilter && matchSearch;
    });

    return filtered.sort((a, b) => {
      if (this.sortMode === 'nama') return a.name.localeCompare(b.name);
      return (b.lastSeenRaw ?? '').localeCompare(a.lastSeenRaw ?? '');
    });
  }

  get semualCount(): number    { return this.people.length; }
  get terdaftarCount(): number { return this.people.filter(p => p.status === 'terdaftar').length; }
  get unknownCount(): number   { return this.people.filter(p => p.status === 'unknown').length; }

  setFilter(f: Filter): void { this.activeFilter = f; }

  toggleSort(): void {
    this.sortMode = this.sortMode === 'terbaru' ? 'nama' : 'terbaru';
  }

  selectPerson(person: Person): void {
    if (this.renamingId !== -1) return;
    this.selectedPerson = person;
    this.previewUrl     = person.thumbnailUrl;
    this.previewMvIdx   = null;
    this.editMode       = false;
    this._fetchMovements(person);
  }

  viewSnapshot(mv: Movement, idx: number, event: Event): void {
    event.stopPropagation();
    if (!mv.snapshot_url) return;
    this.previewMvIdx = idx;
    this.previewUrl   = mv.snapshot_url;
  }

  private _fetchMovements(person: Person): void {
    this.http.get<Movement[]>(
      `${API}/persons/${person.id}/movements?date=${this.selectedDate}`
    ).subscribe({
      next: raw => {
        const mvs = raw.map(mv => ({ ...mv, time: this._fmt(mv.timestamp) }));
        person.movements = mvs;
        if (this.selectedPerson?.id === person.id) {
          this.selectedPerson.movements = mvs;
          // Jangan timpa pilihan manual user
          if (this.previewMvIdx === null) {
            const current = mvs.find(m => m.isCurrent);
            this.previewUrl = current?.snapshot_url ?? person.thumbnailUrl ?? null;
          }
        }
      },
      error: err => console.error('[people] movements error:', err),
    });
  }

  get dateLabel(): string {
    const d = new Date(this.selectedDate + 'T00:00:00');
    return d.toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' });
  }

  onSearch(event: Event): void {
    this.searchQuery = (event.target as HTMLInputElement).value;
  }

  private _todayISO(): string {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  private _fmt(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
  }

  private _fmtFull(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('id-ID', { dateStyle: 'short', timeStyle: 'short' });
  }
}
