import { Pipe, PipeTransform, inject } from '@angular/core';
import { AuthService } from '../services/auth.service';

@Pipe({ name: 'authUrl', standalone: true })
export class AuthUrlPipe implements PipeTransform {
  private auth = inject(AuthService);

  transform(url: string | null | undefined): string | null {
    if (!url) {
      return null;
    }
    const token = this.auth.getToken();
    if (!token) {
      return url;
    }
    // Sisipkan token SEBELUM fragment (#t=... media fragment buat seek video) —
    // kalau ditaruh di belakang, "#" bikin token jadi bagian fragment & auth gagal.
    const h = url.indexOf('#');
    const base = h >= 0 ? url.slice(0, h) : url;
    const frag = h >= 0 ? url.slice(h) : '';
    const sep = base.includes('?') ? '&' : '?';
    return `${base}${sep}token=${encodeURIComponent(token)}${frag}`;
  }
}
